"""
Per-task injection policies for UltimateFlux.

Each class inherits BasePolicy and implements:
  inject_qkv  — attention-level Q/K/V manipulation
  get_block_hooks — transformer block forward hooks (for PFB)
  pre_generate    — one-time setup before the denoising loop

Task coverage:
  NonRigidPolicy             — non-rigid pose/action editing  (§7, Task 1)
  ObjectAdditionPolicy       — insert a new object            (§7, Task 2)
  ObjectReplacementPolicy    — swap an object in place        (§7, Task 3)
  BackgroundReplacePolicy    — regenerate background          (§7, Task 4)
  FineGrainedAttrPolicy      — disentangled attribute edits   (§7, Task 5)
  StylePersonalizationPolicy — reference-image style transfer (§7, Task 7)
  LatentNudgingMixin         — real-image inversion helper    (§7, Task 8)

Not yet implemented:
  GlobalStylePolicy          — null-prompt projection degrades under schnell's guidance distillation
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from typing import Callable, Dict, List, Optional, Tuple

from diffusers.models.embeddings import apply_rotary_emb

from PIL import ImageFilter

from .sampler import BasePolicy, TIER_A, TIER_B, N_DOUBLE, N_SINGLE, N_LAYERS


# ────────────────────────── LAB colour helpers ────────────────────────────────

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) float32 in [0,1] → CIE L*a*b* float32."""
    lin = np.where(rgb > 0.04045,
                   ((rgb + 0.055) / 1.055) ** 2.4,
                   rgb / 12.92).astype(np.float32)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = lin @ M.T
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 2] /= 1.08883
    eps = 0.008856
    f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (903.3 * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE L*a*b* float32 → (H,W,3) float32 in [0,1]."""
    fy = (lab[:, :, 0] + 16.0) / 116.0
    fx = lab[:, :, 1] / 500.0 + fy
    fz = fy - lab[:, :, 2] / 200.0
    eps = 0.206897
    xyz = np.stack([
        np.where(fx > eps, fx ** 3, (116.0 * fx - 16.0) / 903.3),
        np.where(fy > eps, fy ** 3, (116.0 * fy - 16.0) / 903.3),
        np.where(fz > eps, fz ** 3, (116.0 * fz - 16.0) / 903.3),
    ], axis=-1).astype(np.float32)
    xyz[:, :, 0] *= 0.95047
    xyz[:, :, 2] *= 1.08883
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    lin = np.clip(xyz @ M_inv.T, 0.0, None)
    rgb = np.where(lin > 0.0031308,
                   1.055 * lin ** (1.0 / 2.4) - 0.055,
                   12.92 * lin)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _lab_color_transfer(
    src_img: Image.Image,
    ref_img: Image.Image,
    blend_strength: float = 0.92,
    blur_radius: int = 10,
) -> Image.Image:
    """
    Colour-only transfer from ref_img onto src_img, luminance-preserving.

    Both L (luminance) and AB (chromaticity) are transferred inside the
    auto-detected changing region so that large lightness shifts (e.g. black→
    blonde hair) are also captured — but the soft mask keeps the transfer
    localised so surrounding structure is unaffected.

    Region detection: pixels where the two images differ most in full LAB
    distance (Delta-E).  No mask or segmentation required — the difference map
    itself identifies the changing region.
    """
    src = np.array(src_img.convert("RGB")).astype(np.float32) / 255.0
    if ref_img.size != src_img.size:
        ref_img = ref_img.resize(src_img.size, Image.LANCZOS)
    ref = np.array(ref_img.convert("RGB")).astype(np.float32) / 255.0

    src_lab = _rgb_to_lab(src)
    ref_lab = _rgb_to_lab(ref)

    # Full Delta-E distance — captures both hue shifts AND lightness shifts
    delta_e = np.sqrt(((src_lab - ref_lab) ** 2).sum(axis=-1))   # (H, W)
    diff_norm = (delta_e / (delta_e.max() + 1e-8)).astype(np.float32)

    # Smooth the mask (avoids hard colour boundaries at the transfer edge)
    mask_pil = Image.fromarray((diff_norm * 255).astype(np.uint8))
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    soft_mask = np.array(mask_pil).astype(np.float32) / 255.0
    alpha = np.clip(soft_mask * blend_strength, 0.0, 1.0)[:, :, np.newaxis]  # (H,W,1)

    # Blend all three LAB channels inside the mask (colour + lightness change)
    result_lab = (1.0 - alpha) * src_lab + alpha * ref_lab
    result = np.clip(_lab_to_rgb(result_lab) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ─────────────────────────── Shared helpers ───────────────────────────────────

def _image_mask_to_token_mask(
    mask_img: Image.Image,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    """
    Resize a binary PIL mask to FLUX's packed token grid and return a flat
    (1, 1, L_img) float tensor (values 0.0 or 1.0).

    FLUX latent: (1, 16, H//8, W//8)  →  packed 2×2 patch  →  L_img = (H//16)*(W//16).
    """
    th = height // 16
    tw = width  // 16
    mask_np = np.array(mask_img.convert("L").resize((tw, th), Image.NEAREST))
    mask_binary = (mask_np > 127).astype(np.float32)
    return torch.tensor(mask_binary, device=device).reshape(1, 1, th * tw)


def _kv_full_inject(k, v, txt_len: int = 0):
    """Copy source image-token K,V into the edit branch (positions txt_len: only).

    txt_len is the number of text prefix tokens in the current block's sequence.
    For double-stream blocks this equals encoder_hidden_states.shape[1] (e.g. 512).
    For single-stream blocks pass max_sequence_length (the text prefix offset).
    Text tokens are left untouched so each branch preserves its own prompt conditioning.
    This matches FreeFlux's kc_tgt_modified[:,:,512:,:] = kc_src[:,:,512:,:] pattern.
    """
    k_src, k_edit = k.chunk(2)
    v_src, v_edit = v.chunk(2)
    k_edit = k_edit.clone()
    v_edit = v_edit.clone()
    k_edit[:, :, txt_len:, :] = k_src[:, :, txt_len:, :]
    v_edit[:, :, txt_len:, :] = v_src[:, :, txt_len:, :]
    return torch.cat([k_src, k_edit]), torch.cat([v_src, v_edit])


def _k_only_inject(k, v, txt_len: int = 0):
    """Copy source image-token K (only) into the edit branch; V is left untouched.

    Use for colour-change double_stream mode: K injection anchors spatial layout
    (same attention positions → same face/object structure) while V remains
    conditioned on the edit text → full colour freedom without overlay artifacts.
    """
    k_src, k_edit = k.chunk(2)
    k_edit = k_edit.clone()
    k_edit[:, :, txt_len:, :] = k_src[:, :, txt_len:, :]
    return torch.cat([k_src, k_edit]), v


def _masked_kv_inject(k, v, mask_1d: torch.Tensor, txt_len: int):
    """
    Soft-blend source K,V into edit K,V at image-token positions weighted by mask_1d.
    mask_1d: (L_img,) float on the correct device.  1.0 = inject from source.
    """
    k_src, k_edit = k.chunk(2)
    v_src, v_edit = v.chunk(2)
    k_edit = k_edit.clone()
    v_edit = v_edit.clone()
    m = mask_1d.to(k.device)                      # (L_img,)
    m4 = m.view(1, 1, -1, 1)                      # broadcast over (B, H, L, D)
    k_edit[:, :, txt_len:, :] = k_src[:, :, txt_len:, :] * m4 + k_edit[:, :, txt_len:, :] * (1 - m4)
    v_edit[:, :, txt_len:, :] = v_src[:, :, txt_len:, :] * m4 + v_edit[:, :, txt_len:, :] * (1 - m4)
    return torch.cat([k_src, k_edit]), torch.cat([v_src, v_edit])


def _step_active(step: int, n_steps: int, frac: Tuple[float, float]) -> bool:
    return int(frac[0] * n_steps) <= step < int(frac[1] * n_steps)


# ─────────────────── Reasoning processor (object addition) ───────────────────

class _ReasoningAttnProcessor:
    """
    Used during the ObjectAdditionPolicy reasoning pass.

    Runs standard FLUX attention with K,V injection at hotspot layers (same as
    NonRigidPolicy) so the target branch stays anchored to the source layout.

    At derive_step and double-stream hotspot layers, additionally computes
    cross-attention scores from the added-word T5 tokens to image tokens —
    giving us a heatmap of where in the image the new object will appear.
    Only the added-word rows of the attention matrix are computed (memory-efficient).
    """

    def __init__(
        self,
        hotspot_layers: List[int],
        subject_idx: List[int],
        derive_step: int,
        txt_len_single: int = 512,
        total_layers: int = N_LAYERS,
    ):
        self.hotspot_layers  = set(hotspot_layers)
        self.double_hotspots = {l for l in hotspot_layers if l < N_DOUBLE}
        self.subject_idx     = subject_idx
        self.derive_step     = derive_step
        self.txt_len_single  = txt_len_single
        self.total_layers    = total_layers
        self.cur_step        = 0
        self.cur_layer       = 0
        self.object_attn: Optional[torch.Tensor] = None  # (L_img,) accumulated
        self.num_captures    = 0

    def _tick(self):
        self.cur_layer += 1
        if self.cur_layer == self.total_layers:
            self.cur_layer = 0
            self.cur_step += 1

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        layer     = self.cur_layer
        is_double = encoder_hidden_states is not None
        B         = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        inner_dim = k.shape[-1]
        head_dim  = inner_dim // attn.heads

        q = q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # K,V injection at hotspot layers — anchors target to source spatial structure
        if layer in self.hotspot_layers and B >= 2:
            img_offset = txt_len if is_double else self.txt_len_single
            k_src, k_tgt = k.chunk(2)
            v_src, v_tgt = v.chunk(2)
            k_tgt_mod = k_tgt.clone()
            v_tgt_mod = v_tgt.clone()
            k_tgt_mod[:, :, img_offset:, :] = k_src[:, :, img_offset:, :]
            v_tgt_mod[:, :, img_offset:, :] = v_src[:, :, img_offset:, :]
            k = torch.cat([k_src, k_tgt_mod])
            v = torch.cat([v_src, v_tgt_mod])

        # Cross-attention score capture at derive_step, double-stream hotspot layers
        if (self.cur_step == self.derive_step and layer in self.double_hotspots
                and is_double and txt_len > 0 and B >= 2 and self.subject_idx):
            valid = [i for i in self.subject_idx if i < txt_len]
            if valid:
                q_word = q[1:2, :, valid, :].float()  # (1, H, n_word, D)
                k_img  = k[1:2, :, txt_len:, :].float()  # (1, H, L_img, D)
                scores = torch.einsum('bhid,bhjd->bhij', q_word, k_img) * (head_dim ** -0.5)
                probs  = scores.softmax(dim=-1)           # (1, H, n_word, L_img)
                contrib = probs[0].mean(0).sum(0).detach().cpu()  # (L_img,)
                if self.object_attn is None:
                    self.object_attn = contrib
                else:
                    self.object_attn = self.object_attn + contrib
                self.num_captures += 1

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            self._tick()
            return out, enc_out

        self._tick()
        return out


def _get_t5_token_indices(pipe, text: str, word: str) -> List[int]:
    """Return 0-based T5 token indices where word appears in text (no special tokens)."""
    tok = getattr(pipe, 'tokenizer_2', None) or pipe.tokenizer
    prompt_ids = tok(text, add_special_tokens=False).input_ids
    word_ids   = tok(word, add_special_tokens=False).input_ids
    if not word_ids:
        return []
    indices: List[int] = []
    for i in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[i:i + len(word_ids)] == word_ids:
            indices.extend(range(i, i + len(word_ids)))
    if not indices:
        for wt in word_ids:
            indices += [j for j, pt in enumerate(prompt_ids) if pt == wt]
    return sorted(set(indices))


@torch.no_grad()
def _derive_object_mask(
    pipe,
    source_prompt: str,
    target_prompt: str,
    added_word: str,
    seed: int,
    n_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    max_sequence_length: int,
    derive_step: int,
    device: str,
    hotspot_layers: List[int],
    top_k_frac: float = 0.15,
) -> List[int]:
    """
    Run a short B=2 reasoning pass to find where the new object should appear.

    Returns absolute token indices (>= max_sequence_length) corresponding to the
    image-token positions where the added-word T5 tokens have highest cross-attention.
    """
    from diffusers.utils.torch_utils import randn_tensor

    exec_device = getattr(pipe, '_execution_device', device)
    num_ch = pipe.transformer.config.in_channels // 4
    lat_h  = height // 8
    lat_w  = width  // 8

    g   = torch.Generator(device=exec_device).manual_seed(seed)
    one = randn_tensor((1, num_ch, lat_h, lat_w), generator=g,
                       device=exec_device, dtype=torch.bfloat16)
    shared = (one.expand(2, -1, -1, -1).clone()
              .view(2, num_ch, lat_h // 2, 2, lat_w // 2, 2)
              .permute(0, 2, 4, 1, 3, 5)
              .reshape(2, (lat_h // 2) * (lat_w // 2), num_ch * 4))

    subject_idx = _get_t5_token_indices(pipe, target_prompt, added_word)
    if not subject_idx:
        print(f"[UltimateFlux] '{added_word}' not found in T5 tokens — skipping mask derivation")
        return []
    print(f"[UltimateFlux] T5 indices for '{added_word}': {subject_idx}")

    proc = _ReasoningAttnProcessor(
        hotspot_layers=hotspot_layers,
        subject_idx=subject_idx,
        derive_step=derive_step,
        txt_len_single=max_sequence_length,
        total_layers=N_LAYERS,
    )
    pipe.transformer.set_attn_processor(proc)

    # Run only derive_step+1 steps — enough to capture attention at step derive_step
    reasoning_steps = min(n_steps, derive_step + 1)
    print(f"[UltimateFlux] Reasoning pass ({reasoning_steps} steps)…")
    pipe(
        prompt=[source_prompt, target_prompt],
        latents=shared,
        num_inference_steps=reasoning_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        output_type="pil",
    )

    if proc.object_attn is None or proc.num_captures == 0:
        print("[UltimateFlux] Warning: no attention scores captured in reasoning pass")
        return []

    attn_avg = proc.object_attn / proc.num_captures  # (L_img,)
    n_img    = attn_avg.shape[0]
    k        = max(1, int(top_k_frac * n_img))
    rel_idx  = attn_avg.topk(k).indices                  # relative image indices
    abs_idx  = (rel_idx + max_sequence_length).tolist()  # absolute (>= max_sequence_length)
    print(f"[UltimateFlux] Derived {len(abs_idx)} object-region tokens "
          f"(top {top_k_frac*100:.0f}% of {n_img})")
    return abs_idx


# ─────────────────────────── Task 2: Object addition ─────────────────────────

class ObjectAdditionPolicy(BasePolicy):
    """
    Object addition using FreeFlux's layout-aware K,V injection.

    Reasoning pass  (runs inside pre_generate when added_word is given):
        Short B=2 denoising pass that captures cross-attention from the
        added-word T5 tokens to image tokens, identifying WHERE in the scene
        the new object should appear.  Produces derive_idx_list (absolute token
        positions in the combined [text, image] sequence).

    Main generation (the generate_dual_branch call):
        B=2 denoising with [source_prompt, edit_prompt].
        At hotspot layers, source K,V are copied to the edit branch for ALL image
        tokens (freezing the background), then RESTORED for derive_idx_list tokens
        so the new object can emerge there freely.

    If placement_mask is provided, it overrides the automatic mask derivation.
    If neither added_word nor placement_mask is given, source K,V are injected
    everywhere — background is frozen but the new object can't appear.
    """

    # Only the double-stream hotspot layers [1, 2, 4] are used for injection.
    # Using all 7 hotspot layers (including single-stream [26, 30, 54, 55]) anchors
    # the layout too rigidly: freed derive_idx tokens that overlap with existing
    # object boundaries generate a duplicate of the existing object instead of the
    # new one.  Double-stream-only gives enough coarse spatial anchoring while
    # leaving enough compositional freedom for the new object to appear cleanly.
    HOTSPOT_LAYERS = [1, 2, 4]

    def __init__(
        self,
        added_word: Optional[str] = None,
        placement_mask: Optional[Image.Image] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 1.0),
        derive_step: int = 7,
        top_k_frac: float = 0.08,
    ):
        self.added_word        = added_word
        self.raw_mask          = placement_mask
        self.inject_steps_frac = inject_steps_frac
        self.derive_step       = derive_step
        self.top_k_frac        = top_k_frac
        self._derive_idx: Optional[torch.Tensor] = None  # (N,) long on device
        self._token_mask: Optional[torch.Tensor] = None  # (1,1,L_img) float
        self._txt_len_single: int = 512

    def pre_generate(self, pipe, device="cuda", height=1024, width=1024,
                     num_steps=28, seed=0, source_prompt="", edit_prompt="",
                     max_sequence_length=512, guidance_scale=3.5, **kwargs):
        self._txt_len_single = max_sequence_length
        target_prompt = edit_prompt or source_prompt

        if self.raw_mask is not None:
            self._token_mask = _image_mask_to_token_mask(self.raw_mask, height, width, device)
            self._derive_idx = None
        elif self.added_word:
            abs_idx = _derive_object_mask(
                pipe=pipe,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                added_word=self.added_word,
                seed=seed,
                n_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                max_sequence_length=max_sequence_length,
                derive_step=self.derive_step,
                device=device,
                hotspot_layers=self.HOTSPOT_LAYERS,
                top_k_frac=self.top_k_frac,
            )
            if abs_idx:
                self._derive_idx = torch.tensor(abs_idx, dtype=torch.long)
        else:
            print("[UltimateFlux] ObjectAdditionPolicy: no added_word or mask — "
                  "background frozen globally, new object may not appear. "
                  "Pass added_word='<noun>' for automatic mask derivation.")
            self._derive_idx = None

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if layer not in self.HOTSPOT_LAYERS:
            return q, k, v
        if not _step_active(step, n_steps, self.inject_steps_frac):
            return q, k, v

        img_offset = txt_len if layer < N_DOUBLE else self._txt_len_single

        k_src, k_edit = k.chunk(2)
        v_src, v_edit = v.chunk(2)
        k_new = k_edit.clone()
        v_new = v_edit.clone()

        if self._token_mask is not None:
            # User-provided mask: source K,V in background (mask=0), edit K,V in object (mask=1)
            outside  = (1.0 - self._token_mask.squeeze()).to(k.device)   # (L_img,)
            outside4 = outside.view(1, 1, -1, 1)
            inside4  = (1.0 - outside4)
            k_new[:, :, img_offset:, :] = (k_src[:, :, img_offset:, :] * outside4
                                           + k_edit[:, :, img_offset:, :] * inside4)
            v_new[:, :, img_offset:, :] = (v_src[:, :, img_offset:, :] * outside4
                                           + v_edit[:, :, img_offset:, :] * inside4)
        else:
            # Replace all image tokens with source (freeze background)
            k_new[:, :, img_offset:, :] = k_src[:, :, img_offset:, :]
            v_new[:, :, img_offset:, :] = v_src[:, :, img_offset:, :]

            # Restore object-region tokens so the new object can emerge freely
            if self._derive_idx is not None:
                idx = self._derive_idx.to(k.device)
                idx = idx.clamp(0, k.shape[2] - 1)
                k_new[:, :, idx, :] = k_edit[:, :, idx, :]
                v_new[:, :, idx, :] = v_edit[:, :, idx, :]

        return q, torch.cat([k_src, k_new]), torch.cat([v_src, v_new])


# ─────────────────────────── Task 1: Non-rigid ───────────────────────────────

class NonRigidPolicy(BasePolicy):
    """
    Freeze appearance via image-token K,V injection at FreeFlux's content-similarity layers.
    Structure layers (not in inject_layers) are left free, allowing pose/shape changes.

    Layer set defaults to TIER_A — FreeFlux's dev-validated content-similarity layers.
    """

    def __init__(
        self,
        inject_layers: Optional[List[int]] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 1.0),
    ):
        self.inject_layers = inject_layers if inject_layers is not None else TIER_A
        self.inject_steps_frac = inject_steps_frac
        self._txt_len_single = 512  # updated in pre_generate

    def pre_generate(self, pipe, max_sequence_length: int = 512, **kwargs):
        self._txt_len_single = max_sequence_length

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if layer in self.inject_layers and _step_active(step, n_steps, self.inject_steps_frac):
            # txt_len > 0 for double-stream (sampler sets it from encoder_hidden_states).
            # txt_len == 0 for single-stream (no separate encoder_hidden_states), but the
            # hidden_states still carry a text prefix of length _txt_len_single.
            img_offset = txt_len if txt_len > 0 else self._txt_len_single
            k, v = _kv_full_inject(k, v, img_offset)
        return q, k, v


# ─────────────────────────── Task 3: Object replacement ──────────────────────

class ObjectReplacementPolicy(BasePolicy):
    """
    Swap one object for another while keeping the background pixel-identical.

    With mask (best quality):
        Inject source K,V at ALL 57 layers, but ONLY outside the replacement
        mask.  Background (mask=0) is fully frozen at every layer; the object
        region (mask=1) is completely free → new object emerges from edit prompt.

    Without mask (approximate):
        Inject source K,V at TIER_A (content-similarity) ∪ HOTSPOT_LAYERS
        (position-dependent) globally — 20 layers total.  Covers both appearance
        and spatial-layout features, giving better background preservation than
        TIER_A alone.  The remaining 37 free layers let the object identity change
        via Q from the edit branch.

    mask: binary PIL Image — 1 (white) = pixels where the object IS being replaced.
    """

    # Union of content-similarity (TIER_A) and layout-hotspot layers — used for
    # the no-mask case to cover both appearance and positional feature channels.
    _PRESERVE_LAYERS = sorted(set(TIER_A) | {1, 2, 4, 26, 30, 54, 55})

    def __init__(
        self,
        mask: Optional[Image.Image] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 1.0),
    ):
        self.raw_mask = mask
        self.inject_steps_frac = inject_steps_frac
        self._token_mask: Optional[torch.Tensor] = None   # (1,1,L_img)
        self._txt_len_single = 512  # updated in pre_generate

    def pre_generate(self, pipe, device="cuda", height=1024, width=1024,
                     max_sequence_length=512, **kwargs):
        if self.raw_mask is not None:
            self._token_mask = _image_mask_to_token_mask(self.raw_mask, height, width, device)
        self._txt_len_single = max_sequence_length

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if not _step_active(step, n_steps, self.inject_steps_frac):
            return q, k, v

        img_offset = txt_len if txt_len > 0 else self._txt_len_single

        if self._token_mask is not None:
            # Masked: freeze background at ALL layers; object region is fully free.
            # (Previous bug: TIER_A was injected globally, freezing object appearance
            # even inside the mask and preventing proper object replacement.)
            outside = (1.0 - self._token_mask.squeeze(0).squeeze(0))  # (L_img,) bg=1
            k, v = _masked_kv_inject(k, v, outside, img_offset)
        else:
            # No mask: inject only at HOTSPOT_LAYERS (position-dependent, 7 layers).
            # Preserves spatial layout / composition so the background doesn't drift.
            # TIER_A (appearance layers) is intentionally LEFT FREE so the object's
            # texture and colour can change (apple→orange, wood→metal).
            # Injecting TIER_A here would freeze the object's appearance and prevent
            # the replacement from taking effect.
            if layer in {1, 2, 4, 26, 30, 54, 55}:
                k, v = _kv_full_inject(k, v, img_offset)

        return q, k, v


# ─────────────────────────── Task 4: Background replacement ──────────────────

@torch.no_grad()
def _generate_source_preview(
    pipe,
    source_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    device: str,
) -> Optional[Image.Image]:
    """
    Generate the source image with standard attention using the same seed/latent
    that generate_dual_branch will use for its source branch.  The result is
    pixel-identical to what the source branch produces in the dual-branch loop,
    so any mask derived from it transfers directly.
    """
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.models.attention_processor import FluxAttnProcessor2_0

    try:
        exec_device = getattr(pipe, '_execution_device', device)
        num_ch = pipe.transformer.config.in_channels // 4
        lat_h, lat_w = height // 8, width // 8

        g   = torch.Generator(device=exec_device).manual_seed(seed)
        one = randn_tensor((1, num_ch, lat_h, lat_w), generator=g,
                           device=exec_device, dtype=torch.bfloat16)
        latent = (one.view(1, num_ch, lat_h // 2, 2, lat_w // 2, 2)
                  .permute(0, 2, 4, 1, 3, 5)
                  .reshape(1, (lat_h // 2) * (lat_w // 2), num_ch * 4))

        pipe.transformer.set_attn_processor(FluxAttnProcessor2_0())
        result = pipe(
            prompt=source_prompt,
            latents=latent,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            output_type="pil",
        )
        return result.images[0]
    except Exception as exc:
        print(f"[UltimateFlux] Source preview generation failed: {exc}")
        return None


def _auto_fg_mask_sam2(
    source_image: Image.Image,
    device: str = "cuda",
    sam2_model_id: str = "facebook/sam2-hiera-large",
) -> Optional[Image.Image]:
    """
    Run SAM2 automatic mask generation on source_image.
    Returns a binary PIL mask (white = foreground subject) or None.

    Requires: pip install sam2
    Model is downloaded automatically from HuggingFace on first use.
    """
    try:
        import numpy as np
        from sam2.build_sam import build_sam2_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("[UltimateFlux] SAM2 not installed. Run: pip install sam2")
        return None

    try:
        sam2 = build_sam2_hf(sam2_model_id, device=device)
        generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )

        img_np = np.array(source_image.convert("RGB"))
        masks  = generator.generate(img_np)

        if not masks:
            print("[UltimateFlux] SAM2 found no masks in the source image.")
            return None

        # Score each mask: prefer high stability + centred in the image
        H, W = img_np.shape[:2]
        cx, cy = W / 2.0, H / 2.0

        def _score(m):
            bx = m['bbox'][0] + m['bbox'][2] / 2.0
            by = m['bbox'][1] + m['bbox'][3] / 2.0
            dist_norm = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / max(W, H)
            return m['stability_score'] - dist_norm

        best = max(masks, key=_score)
        fg   = (best['segmentation'].astype(np.uint8) * 255)
        return Image.fromarray(fg)

    except Exception as exc:
        print(f"[UltimateFlux] SAM2 segmentation failed: {exc}")
        import traceback; traceback.print_exc()
        return None


class BackgroundReplacePolicy(BasePolicy):
    """
    Regenerate the background while preserving the foreground subject.

    Mask source (priority order):
      1. fg_mask  — user-supplied PIL mask (white = foreground to keep).
      2. SAM2     — automatic segmentation of a source preview image (same seed
                    as the main generation so the mask aligns exactly).
                    Enabled when use_sam2=True (default) and SAM2 is installed.
      3. Fallback — TIER_A K,V injection globally; subject is approximately
                    preserved without a precise boundary.

    Masked path (options 1 & 2, FreeFlux approach):
        Value-only injection inside the fg region at all 57 layers.
        Subject V values from source are blended in; background V is free.

    sam2_model_id: HuggingFace model ID for SAM2 (default: sam2-hiera-large).
    """

    def __init__(
        self,
        fg_mask: Optional[Image.Image] = None,
        use_sam2: bool = True,
        sam2_model_id: str = "facebook/sam2-hiera-large",
    ):
        self.raw_mask      = fg_mask
        self.use_sam2      = use_sam2
        self.sam2_model_id = sam2_model_id
        self._token_mask: Optional[torch.Tensor] = None
        self._txt_len_single = 512

    def pre_generate(self, pipe, device="cuda", height=1024, width=1024,
                     num_steps=28, seed=0, source_prompt="",
                     guidance_scale=3.5, max_sequence_length=512, **kwargs):
        self._txt_len_single = max_sequence_length

        if self.raw_mask is not None:
            # User-supplied mask — highest priority
            self._token_mask = _image_mask_to_token_mask(
                self.raw_mask, height, width, device)
            return

        if self.use_sam2:
            print("[UltimateFlux] BackgroundReplacePolicy: generating source preview for SAM2…")
            src_img = _generate_source_preview(
                pipe, source_prompt, seed, height, width,
                num_steps, guidance_scale, max_sequence_length, device,
            )
            if src_img is not None:
                print("[UltimateFlux] Running SAM2 automatic segmentation…")
                fg_mask = _auto_fg_mask_sam2(
                    src_img, device=device, sam2_model_id=self.sam2_model_id)
                if fg_mask is not None:
                    self._token_mask = _image_mask_to_token_mask(
                        fg_mask, height, width, device)
                    print("[UltimateFlux] SAM2 foreground mask ready.")
                    return
            print("[UltimateFlux] SAM2 unavailable — falling back to TIER_A global injection.")

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        img_offset = txt_len if txt_len > 0 else self._txt_len_single

        if self._token_mask is not None:
            # Masked (FreeFlux approach): value-only injection inside foreground.
            v_src, v_edit = v.chunk(2)
            v_edit = v_edit.clone()
            fg  = self._token_mask.squeeze(0).squeeze(0).to(v.device)  # (L_img,)
            fg4 = fg.view(1, 1, -1, 1)
            v_edit[:, :, img_offset:, :] = (
                v_src[:, :, img_offset:, :] * fg4 + v_edit[:, :, img_offset:, :] * (1 - fg4)
            )
            v = torch.cat([v_src, v_edit])
        else:
            # Fallback (no mask): K,V injection at TIER_A globally.
            if layer in TIER_A:
                k, v = _kv_full_inject(k, v, img_offset)

        return q, k, v


# ─────────────────────────── Task 5: Fine-grained attribute ──────────────────

class FineGrainedAttrPolicy(BasePolicy):
    """
    Identity-preserving attribute editing.  Three injection modes via inject_layers:

    None (default) → _PRESERVE_LAYERS (TIER_A ∪ HOTSPOT, 20 layers).
        Tightest identity lock — for ADDING an attribute (glasses).

    list(range(N_DOUBLE)) → double-stream blocks only (layers 0-18).
        All 19 joint text-image blocks are locked (same identity, same scene).
        All 38 single-stream blocks are FREE — text drives colour there.
        Use for COLOUR changes (hair colour, car colour): identity stays,
        colour is rendered freely in the single-stream refinement stage.

    TIER_A → Appearance locked, position flexible — BREED / shape changes.
    """
    _PRESERVE_LAYERS = sorted(set(TIER_A) | {1, 2, 4, 26, 30, 54, 55})  # 20 layers
    _DOUBLE_STREAM   = list(range(N_DOUBLE))                               # 0-18

    def __init__(
        self,
        inject_layers: Optional[List[int]] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 1.0),
        key_only: bool = False,
    ):
        self._inject_layers = (
            inject_layers if inject_layers is not None else self._PRESERVE_LAYERS
        )
        self.inject_steps_frac = inject_steps_frac
        self.key_only = key_only
        self._txt_len_single = 512

    def pre_generate(self, pipe, max_sequence_length: int = 512, **kwargs):
        self._txt_len_single = max_sequence_length

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if layer not in self._inject_layers:
            return q, k, v
        if not _step_active(step, n_steps, self.inject_steps_frac):
            return q, k, v

        img_offset = txt_len if txt_len > 0 else self._txt_len_single
        if self.key_only:
            k, v = _k_only_inject(k, v, img_offset)
        else:
            k, v = _kv_full_inject(k, v, img_offset)
        return q, k, v


# ─────────────────────────── Task 5b: ColorCtrl colour editing ───────────────

class ColorCtrlPolicy(BasePolicy):
    """
    Faithful implementation of ColorCtrl (arXiv:2508.09131) for FLUX.1-dev.

    Scope — SINGLE-STREAM BLOCKS ONLY (layers 19-56):
        Double-stream blocks (0-18): standard SDPA, no modification.
        Single-stream blocks: manual head-chunked attention with:

    Structure Preservation (§3.3):
        Before softmax, the source branch's image-to-image (v-v) attention
        score quadrant is copied into the target branch.  This forces the
        target to attend to the SAME spatial positions as the source, locking
        the geometric layout while allowing colour to change via V + text.

    Color Preservation (§3.4):
        A binary preserve_mask identifies the NON-editing region (background,
        face, licence plate, …).  Source V^image is blended into the target's
        value tensor at those positions before the attention is computed, so
        those regions are anchored to the source appearance.  Editing-region
        tokens (hair, car body) keep the target's V and receive the new colour.

    Mask derivation:
        At the FIRST single-stream attention call (step=0, layer=19), the target
        branch's image→text attention scores are used to identify which image
        tokens attend most strongly to the target-prompt text (or to the
        specific colour-word tokens when color_word is supplied).  The top-K%
        tokens become the editing region (preserve_mask=0); all others are
        preserved (preserve_mask=1).  The mask is cached and reused for all
        subsequent layers and steps.

    Parameters
    ----------
    top_k_frac   : Fraction of image tokens treated as the editing region.
                   0.2 = 20%.  Increase for larger edits, decrease to tighten.
    qk_frac      : Fraction of denoising steps for structure preservation
                   (v-v score injection).  1.0 = all steps.
    v_frac       : Fraction of denoising steps for colour preservation
                   (V masking).  1.0 = all steps.
    color_word   : The specific colour word in the edit prompt (e.g. "blonde",
                   "blue").  When supplied, the mask focuses on tokens that
                   attend to its T5 token IDs, giving a tighter editing region.
    chunk_size   : Heads processed at once in the manual attention loop.
                   Reduce to 2 or 1 if you encounter OOM errors.
    """

    def __init__(
        self,
        top_k_frac: float = 0.2,
        qk_frac: float = 1.0,
        v_frac: float = 1.0,
        color_word: Optional[str] = None,
        chunk_size: int = 4,
        mask_build_step: int = 5,
        reweight_scale: float = 1.0,
        ds_key_inject: bool = False,
    ):
        self.top_k_frac      = top_k_frac
        self.qk_frac         = qk_frac
        self.v_frac          = v_frac
        self.color_word      = color_word
        self.chunk_size      = chunk_size
        self.mask_build_step = mask_build_step
        # §3.5 Attribute re-weighting: multiply image→colour-word attention scores
        # by reweight_scale before softmax.  Image tokens that naturally attend to
        # "blonde"/"blue" pull even more from V_txt[color_word] → stronger colour.
        # Applied in ALL single-stream layers, independently of the V-mask.
        # Set reweight_scale=1.0 to disable (paper: unspecified value, typically 2–5).
        self.reweight_scale  = reweight_scale
        # K-only injection in double-stream blocks (0-18): locks spatial layout via K
        # without locking V, so face structure is preserved while colour changes freely.
        # Use with qk_frac=0, v_frac=0, reweight_scale≥2 for cleanest colour editing.
        self.ds_key_inject   = ds_key_inject
        self._txt_len        = 512
        self._color_tok_ids: List[int] = []
        self._mask: Optional[torch.Tensor] = None   # (n_img,) preserve mask
        self._mask_built     = False

    def pre_generate(
        self,
        pipe,
        max_sequence_length: int = 512,
        edit_prompt: str = "",
        **kwargs,
    ):
        self._txt_len = max_sequence_length
        self._mask = None
        self._mask_built = False
        self._color_tok_ids = []
        if self.color_word and edit_prompt:
            self._color_tok_ids = _get_t5_token_indices(pipe, edit_prompt, self.color_word)
            print(f"[ColorCtrl] '{self.color_word}' → T5 indices: {self._color_tok_ids}")

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        # Double-stream K injection: locks spatial layout (same face positions)
        # without locking V, so single-stream can still drive colour change freely.
        if self.ds_key_inject and txt_len > 0:
            img_offset = txt_len
            k_src, k_tgt = k.chunk(2)
            k_tgt_new = k_tgt.clone()
            k_tgt_new[:, :, img_offset:, :] = k_src[:, :, img_offset:, :]
            k = torch.cat([k_src, k_tgt_new])
        return q, k, v

    def inject_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        step: int,
        n_steps: int,
        txt_len: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Custom attention for single-stream blocks only.

        Double-stream blocks (txt_len > 0): returns None → standard SDPA.
        Single-stream blocks: manual head-chunked attention with:
          • V masking (colour preservation, gated by v_frac)
          • v-v pre-softmax score injection (structure preservation, gated by qk_frac)
        """
        # ── Double-stream: leave unchanged (paper does standard SDPA there) ────
        if txt_len > 0:
            return None

        B = q.shape[0]
        if B < 2:
            return None

        text_len = self._txt_len
        n_total  = q.shape[2]
        if n_total <= text_len:
            return None

        n_img    = n_total - text_len
        n_heads  = q.shape[1]
        head_dim = q.shape[-1]
        scale    = head_dim ** -0.5

        do_qk = step < int(self.qk_frac * n_steps)  # v-v score injection active?
        do_v  = step < int(self.v_frac  * n_steps)  # V masking active?
        do_rw = (self.reweight_scale > 1.0          # re-weighting active when:
                 and bool(self._color_tok_ids))      #   scale set AND color_word known

        # ── Mask build: attempt once conditions are met (needed for do_v / do_qk) ─
        # Re-weighting (do_rw) does NOT need the mask — it fires from step 0.
        # The mask is only built when mask_build_step is reached, giving the model
        # time to form spatial structure before localising the editing region.
        if not self._mask_built and step >= self.mask_build_step and layer >= N_DOUBLE:
            q_tgt    = q[B // 2:]
            k_tgt_tx = k[B // 2:, :, :text_len, :]
            v2t      = torch.matmul(
                q_tgt[:, :, text_len:, :].float(),
                k_tgt_tx.float().transpose(-2, -1),
            ) * scale                                # (1, H, n_img, n_txt)
            v2t_mean = v2t.mean(dim=(0, 1))          # (n_img, n_txt)
            if self._color_tok_ids:
                ids        = [i for i in self._color_tok_ids if i < text_len]
                edit_score = (v2t_mean[:, ids].mean(dim=-1)
                              if ids else v2t_mean.mean(dim=-1))
            else:
                edit_score = v2t_mean.mean(dim=-1)
            k_top        = max(1, int(self.top_k_frac * n_img))
            edit_idx     = edit_score.topk(k_top).indices
            preserve_mask = torch.ones(n_img, device=q.device, dtype=torch.float32)
            preserve_mask[edit_idx] = 0.0           # 0 = editing region
            self._mask       = preserve_mask
            self._mask_built = True
            print(f"[ColorCtrl] mask built layer={layer} step={step}: "
                  f"{k_top}/{n_img} editing tokens "
                  f"({'colour-word focused' if self._color_tok_ids else 'all-text mean'})")

        v_active    = do_v and self._mask is not None
        need_manual = do_qk or v_active or do_rw
        if not need_manual:
            return None

        # ── §3.4 Colour Preservation: V masking ──────────────────────────────
        if v_active:
            v_src, v_tgt = v.chunk(2)
            mask4 = self._mask.to(device=v.device, dtype=v.dtype).view(1, 1, -1, 1)
            v_tgt_new = v_tgt.clone()
            v_tgt_new[:, :, text_len:, :] = (
                v_src[:, :, text_len:, :] * mask4
                + v_tgt[:, :, text_len:, :] * (1.0 - mask4)
            )
            v = torch.cat([v_src, v_tgt_new])

        # ── Fast path: V masking only, no pre-softmax score manipulation ─────
        if not do_qk and not do_rw:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # ── Manual attention: §3.3 v-v injection and/or §3.5 re-weighting ────
        out      = torch.zeros(B, n_heads, n_total, head_dim,
                               device=q.device, dtype=q.dtype)
        rw_ids   = ([i for i in self._color_tok_ids if i < text_len]
                    if do_rw else [])

        for h0 in range(0, n_heads, self.chunk_size):
            h1 = min(h0 + self.chunk_size, n_heads)
            q_c, k_c, v_c = q[:, h0:h1], k[:, h0:h1], v[:, h0:h1]

            # Source branch — standard attention; save v-v scores if needed
            q_s, k_s = q_c[:B // 2], k_c[:B // 2]
            scores_s = torch.matmul(q_s.float(), k_s.float().transpose(-2, -1)) * scale
            vv_src   = scores_s[:, :, text_len:, text_len:].clone() if do_qk else None
            probs_s  = torch.softmax(scores_s, dim=-1).to(q.dtype)
            out[:B // 2, h0:h1] = torch.matmul(probs_s, v_c[:B // 2])

            # Target branch
            q_t, k_t = q_c[B // 2:], k_c[B // 2:]
            scores_t  = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) * scale

            # §3.3 Structure: replace image→image pre-softmax scores with source's
            if do_qk and vv_src is not None:
                scores_t[:, :, text_len:, text_len:] = vv_src

            # §3.5 Attribute re-weighting: amplify image→colour-word scores.
            # scores_t rows text_len: = image queries; cols rw_ids = colour-word keys.
            # After softmax, image tokens pull proportionally more from V_txt[colour],
            # driving strong colour change without needing a binary mask.
            if rw_ids:
                scores_t[:, :, text_len:, rw_ids] = (
                    scores_t[:, :, text_len:, rw_ids] * self.reweight_scale
                )

            probs_t = torch.softmax(scores_t, dim=-1).to(q.dtype)
            out[B // 2:, h0:h1] = torch.matmul(probs_t, v_c[B // 2:])

        return out


# ─────────────────────────── Task 7: Style personalization ───────────────────

# Pivotal layer for PFB+SAC: double-stream block 1.
# From schnell ablation (§5 of Pipeline_Plan.md): object (0.555), texture (0.51),
# style (0.50) all peak there simultaneously — the same signature SVD-Style used
# to identify Infinity's F₃. Block 1 is a strong starting hypothesis for dev too.
_PIVOTAL_LAYER = 1


def _style_extractor(h: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """SVD with exponential spectral reweighting.  h: (B, L, C) → same shape."""
    B, L, C = h.shape
    results = []
    for b in range(B):
        U, S, Vh = torch.linalg.svd(h[b].float(), full_matrices=False)
        w = torch.exp(-alpha * torch.arange(S.shape[0], device=h.device, dtype=S.dtype))
        results.append((U * (S * w).unsqueeze(0)) @ Vh)
    return torch.stack(results).to(h.dtype)


def _apply_pfb(h_gen: torch.Tensor, h_sty: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """PFB blending: Φ(h_sty) + (h_gen − Φ(h_gen)).  All tensors (1, L, C)."""
    return _style_extractor(h_sty, alpha) + (h_gen - _style_extractor(h_gen, alpha))


@torch.no_grad()
def _extract_style_hidden_states(
    pipe,
    style_image: Image.Image,
    device: str = "cuda",
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 28,
    seed: int = 0,
) -> Optional[torch.Tensor]:
    """
    Extract pivotal-layer hidden states for a style reference image.

    Strategy: encode style image → FLUX latent → add mid-trajectory noise →
    one transformer forward pass (empty-prompt) → capture block 1 output.

    Returns h_sty: (1, L_img, C) float32, or None on failure.
    """
    try:
        exec_device = getattr(pipe, '_execution_device', device)

        # ── Encode style image with VAE ──────────────────────────────────────
        img_t = to_tensor(style_image.resize((width, height))).unsqueeze(0)
        img_t = (img_t * 2.0 - 1.0).to(exec_device, dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(img_t).latent_dist.mean
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        # Pack latents: (B, C, lat_h, lat_w) → (B, lat_h//2 * lat_w//2, C*4)
        B, C, H, W = latents.shape
        latents = (
            latents.view(B, C, H // 2, 2, W // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, (H // 2) * (W // 2), C * 4)
        )

        # Add noise at t=0.5 (mid-trajectory for flow matching)
        noise = torch.randn_like(latents)
        latents_noisy = 0.5 * latents + 0.5 * noise

        # ── Encode empty prompt ──────────────────────────────────────────────
        enc_result = pipe.encode_prompt(
            prompt="",
            prompt_2=None,
            device=exec_device,
            num_images_per_prompt=1,
        )
        # encode_prompt returns (pemb, ppooled) in newer diffusers,
        # or (pemb, ppooled, text_ids) in some older versions.
        pemb, ppooled = enc_result[0], enc_result[1]

        # ── Build RoPE ids ───────────────────────────────────────────────────
        lh, lw = height // 8, width // 8
        img_ids = torch.zeros(lh // 2, lw // 2, 3, device=exec_device, dtype=latents.dtype)
        img_ids[..., 1] = torch.arange(lh // 2, device=exec_device)[:, None]
        img_ids[..., 2] = torch.arange(lw // 2, device=exec_device)[None, :]
        img_ids = img_ids.reshape((lh // 2) * (lw // 2), 3)  # (L_img, 3) — no batch dim
        txt_ids = torch.zeros(pemb.shape[1], 3, device=exec_device, dtype=latents.dtype)

        # ── Guidance tensor (FLUX.1-dev requires it; schnell ignores it) ─────
        guidance = None
        if getattr(pipe.transformer.config, 'guidance_embeds', False):
            guidance = torch.full([1], 3.5, device=exec_device, dtype=latents.dtype)

        # ── Forward hook on pivotal block ────────────────────────────────────
        captured: Dict = {}

        def _hook(module, inp, out):
            # FluxTransformerBlock returns (encoder_hidden_states, hidden_states).
            # Index 1 = image hidden states — the tensor that carries visual/style info.
            h = (out[1] if (isinstance(out, tuple) and len(out) > 1)
                 else (out[0] if isinstance(out, tuple) else out))
            captured["h_sty"] = h.detach().float()

        handle = pipe.transformer.transformer_blocks[_PIVOTAL_LAYER].register_forward_hook(_hook)

        # Use t=500 (mid-trajectory) directly — avoids scheduler.set_timesteps(mu=...)
        # requirement introduced in newer diffusers for dev's dynamic timestep shifting.
        # The transformer receives timestep / 1000 = 0.5, matching latents_noisy (50/50 mix).
        t = torch.tensor([500.0], device=exec_device, dtype=latents.dtype)

        try:
            _ = pipe.transformer(
                hidden_states=latents_noisy.to(pipe.transformer.dtype),
                timestep=t / 1000.0,
                guidance=guidance,
                pooled_projections=ppooled.to(pipe.transformer.dtype),
                encoder_hidden_states=pemb.to(pipe.transformer.dtype),
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )
        finally:
            handle.remove()

        return captured.get("h_sty")  # (1, L_img, C)

    except Exception as exc:
        print(f"[UltimateFlux] Style extraction failed: {exc}")
        import traceback; traceback.print_exc()
        return None


class StylePersonalizationPolicy(BasePolicy):
    """
    Reference-image style transfer on FLUX.1-dev.

    Adapts SVD-Style (Paper 4) to FLUX's continuous hidden-state space at
    double-stream layer 1 — the pivotal layer where object (0.555), texture
    (0.51), and style (0.50) peak simultaneously (§5/§7 of Pipeline_Plan.md).

    PFB (Principal Feature Blending):
        Applied via forward hook on block 1 output.
        h_edit ← Φ(h_sty, α) + (h_edit − Φ(h_edit, α))
        where Φ = SVD with exponential spectral reweighting.

    SAC (Structural Attention Correction):
        Applied via attention processor at block 1.
        Q[edit] ← Q[src],  K[edit] ← K[src]   (preserve spatial structure)
    """

    def __init__(
        self,
        style_image: Optional[Image.Image] = None,
        alpha: float = 1.0,
        sac_steps_frac: Tuple[float, float] = (0.0, 1.0),
        # PFB applied only in the first quarter of denoising — analogous to
        # step 3/12 in SVD-Style (Infinity). Applying at all steps over-injects
        # style features and creates noise in the output.
        pfb_steps_frac: Tuple[float, float] = (0.0, 0.25),
    ):
        self.style_image = style_image
        self.alpha = alpha
        self.sac_steps_frac = sac_steps_frac
        self.pfb_steps_frac = pfb_steps_frac
        self._h_sty: Optional[torch.Tensor] = None
        self._step_counter = [0]     # mutable ref shared with closure
        self._n_steps = 28           # updated in pre_generate

    def pre_generate(self, pipe, device="cuda", height=1024, width=1024,
                     num_steps=28, seed=0, style_image=None, **kwargs):
        self._n_steps = num_steps
        img = style_image if style_image is not None else self.style_image
        if img is not None:
            print("[UltimateFlux] Extracting style features from reference image…")
            self._h_sty = _extract_style_hidden_states(
                pipe, img,
                device=device, height=height, width=width,
                num_steps=num_steps, seed=seed,
            )
            if self._h_sty is not None:
                print(f"[UltimateFlux] Style features captured, shape: {self._h_sty.shape}")
        self._step_counter[0] = 0

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        """SAC: copy image-token Q, K from source branch to edit branch at pivotal layer.

        Text tokens (0:txt_len) are left as-is — both branches share the same prompt
        for style tasks, so text Q,K are already identical; the clone is just explicit.
        Only image tokens (txt_len:) carry visual structure worth preserving.
        """
        if layer != _PIVOTAL_LAYER:
            return q, k, v
        if not _step_active(step, n_steps, self.sac_steps_frac):
            return q, k, v

        q_src, q_edit = q.chunk(2)
        k_src, k_edit = k.chunk(2)
        q_edit = q_edit.clone()
        k_edit = k_edit.clone()
        q_edit[:, :, txt_len:, :] = q_src[:, :, txt_len:, :]
        k_edit[:, :, txt_len:, :] = k_src[:, :, txt_len:, :]
        q = torch.cat([q_src, q_edit])
        k = torch.cat([k_src, k_edit])
        return q, k, v

    def get_block_hooks(self) -> Dict[int, Callable]:
        """PFB: blend style features into block 1 hidden states."""
        h_sty        = self._h_sty
        alpha        = self.alpha
        pfb_frac     = self.pfb_steps_frac
        step_counter = self._step_counter
        n_steps      = self._n_steps

        def _pfb_hook(module, inp, out):
            # Gate: only apply PFB within the configured step window.
            # step_counter tracks how many times this hook has fired = current step.
            cur_step = step_counter[0]
            step_counter[0] += 1

            if h_sty is None:
                return out

            # FluxTransformerBlock returns (encoder_hidden_states, hidden_states).
            is_tuple = isinstance(out, tuple)
            has_two  = is_tuple and len(out) >= 2
            h = out[1] if has_two else (out[0] if is_tuple else out)

            if h.shape[0] < 2 or not _step_active(cur_step, n_steps, pfb_frac):
                return out

            h_edit  = h[1:2]                                 # (1, L_img, C)
            h_style = h_sty.to(h.device, dtype=h.dtype)
            h_new   = _apply_pfb(h_edit, h_style, alpha)
            h_out   = torch.cat([h[0:1], h_new], dim=0)

            if has_two:
                return (out[0], h_out)   # keep text unchanged, replace image hidden states
            return (h_out,) + out[1:] if is_tuple else h_out

        return {_PIVOTAL_LAYER: _pfb_hook}


# ─────────────────────────── Task 8: Real-image inversion helper ─────────────

class LatentNudgingMixin:
    """
    Mixin providing latent nudging (StableFlow §3) for real-image editing.

    Before calling generate_dual_branch, call prepare_inverted_latent() to
    get the nudged+inverted starting latent, then pass it as latents= to pipe().

    NOTE: FLUX ODE inversion is not implemented here — this nudging corrects
    the systematic magnitude drift in DDIM-style inversion of flow-matching
    models (λ=1.15 from StableFlow ablation Table 3).
    """

    NUDGE_LAMBDA = 1.15

    @torch.no_grad()
    def encode_real_image(
        self,
        pipe,
        image: Image.Image,
        height: int = 1024,
        width: int = 1024,
        device: str = "cuda",
        nudge: bool = True,
    ) -> torch.Tensor:
        """
        Encode a real image to FLUX packed latents (with optional latent nudging).
        The returned tensor can be passed as the starting latent for generation.
        """
        img_t = to_tensor(image.resize((width, height))).unsqueeze(0)
        img_t = (img_t * 2.0 - 1.0).to(device, dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(img_t).latent_dist.mean
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        if nudge:
            latents = latents * self.NUDGE_LAMBDA

        # Pack: (B, C, H, W) → (B, H/2*W/2, C*4)
        B, C, H, W = latents.shape
        latents = (
            latents.view(B, C, H // 2, 2, W // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, (H // 2) * (W // 2), C * 4)
        )
        return latents
