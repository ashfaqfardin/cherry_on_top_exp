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

def _freq_aware_k_blend(k: torch.Tensor, img_offset: int, s_hf: float, s_lf: float) -> torch.Tensor:
    """
    Blend source image-token K into edit K with frequency-aware per-dimension weights.

    Approximates Untwisting RoPE (arXiv:2602.05013) without requiring de-rotation:
    After RoPE is applied, the head dimension d encodes RoPE frequency — low d
    (pairs 0,1) are highest-frequency (most spatially sensitive), high d (pairs
    D-2,D-1) are lowest-frequency (most content/semantically sensitive).

    By blending 0% at high-freq dims and s_lf% at low-freq dims, the edit
    branch's attention at non-TIER_A layers is guided by source *content* (colour,
    texture) rather than source *position* — avoiding spatial locking while still
    pulling colour from source.

    s_hf: blend weight at high-freq dimensions (0 = no injection → no spatial lock)
    s_lf: blend weight at low-freq dimensions (partial source → semantic guidance)
    """
    k_src, k_edit = k.chunk(2)
    D = k_src.shape[-1]
    d = torch.arange(D, device=k_src.device, dtype=k_src.dtype)
    # pair_norm: 0.0 at d=0,1 (highest freq), 1.0 at d=D-2,D-1 (lowest freq)
    pair_norm = (d // 2) / max(D // 2 - 1, 1)
    w = (s_hf + (s_lf - s_hf) * pair_norm).view(1, 1, 1, D)
    k_edit = k_edit.clone()
    k_edit[:, :, img_offset:, :] = (
        w * k_src[:, :, img_offset:, :]
        + (1.0 - w) * k_edit[:, :, img_offset:, :]
    )
    return torch.cat([k_src, k_edit])


def _reinhard_color_transfer(edit_img: Image.Image, src_img: Image.Image) -> Image.Image:
    """
    Transfer source image colour statistics onto edit image (Reinhard et al.).

    Matches the per-channel mean and std of edit's LAB representation to source's.
    Preserves the edit image's spatial structure (flying pose) while correcting the
    colour to match the source (bird/cat species colour).

    Used as post-processing for NonRigidPolicy — does NOT affect the attention
    mechanism, so it cannot cause cascade collapse or identical-output failures.
    """
    src  = np.array(src_img.convert("RGB")).astype(np.float32) / 255.0
    edit = np.array(edit_img.convert("RGB")).astype(np.float32) / 255.0
    src_lab  = _rgb_to_lab(src)
    edit_lab = _rgb_to_lab(edit)
    result_lab = edit_lab.copy()
    for c in range(3):
        s_mean = src_lab[..., c].mean()
        s_std  = src_lab[..., c].std() + 1e-6
        e_mean = edit_lab[..., c].mean()
        e_std  = edit_lab[..., c].std() + 1e-6
        result_lab[..., c] = (edit_lab[..., c] - e_mean) * (s_std / e_std) + s_mean
    rgb = np.clip(_lab_to_rgb(result_lab), 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _freq_identity_guidance(
    latents: torch.Tensor,
    lat_h: int, lat_w: int, C: int,
    strength: float,
    low_freq_cutoff: float,
) -> torch.Tensor:
    """
    Frequency-selective identity guidance for non-rigid editing.

    Unpacks FLUX packed latents (B=2, S, C_packed) to spatial (B, C, H, W),
    applies FFT, blends only the low-frequency components of the SOURCE latent
    into the EDIT latent by `strength`, then repacks.

    Low-frequency components encode global colour and smooth gradients.
    High-frequency components (pose edges, fine texture) are untouched.

    The linear blend cos_w/sin_w interpolation in SynPS handles TIER_A content
    alignment; this function corrects the POST-STEP latent in the frequency domain,
    giving a complementary colour anchor that accumulates across denoising steps.
    """
    B = latents.shape[0]
    dtype = latents.dtype

    # Unpack: (B, S, C_packed) → (B, C, lat_h, lat_w)
    spatial = (
        latents.float()
        .view(B, lat_h // 2, lat_w // 2, C, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(B, C, lat_h, lat_w)
    )
    z_src  = spatial[0:1]   # (1, C, H, W)
    z_edit = spatial[1:2]

    fft_src  = torch.fft.rfft2(z_src)
    fft_edit = torch.fft.rfft2(z_edit)

    # Cutoff in FFT-output dims: rfft2 gives (H, W//2+1)
    h_cut = max(1, int(lat_h * low_freq_cutoff))
    w_cut = max(1, int((lat_w // 2 + 1) * low_freq_cutoff))

    fft_edit = fft_edit.clone()
    fft_edit[:, :, :h_cut, :w_cut] = (
        (1.0 - strength) * fft_edit[:, :, :h_cut, :w_cut]
        + strength       * fft_src[:, :, :h_cut, :w_cut]
    )

    z_edit_new = torch.fft.irfft2(fft_edit, s=(lat_h, lat_w))
    spatial_new = torch.cat([z_src, z_edit_new], dim=0).to(dtype)

    # Repack: (B, C, lat_h, lat_w) → (B, S, C_packed)
    return (
        spatial_new
        .view(B, C, lat_h // 2, 2, lat_w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, (lat_h // 2) * (lat_w // 2), C * 4)
    )


class NonRigidPolicy(BasePolicy):
    """
    K,V injection for non-rigid pose/action editing — FreeFlux + SynPS.

    Core mechanism (FreeFlux, ICCV 2025):
    ──────────────────────────────────────
    TIER_A K+V injection (13 content-similarity layers): Q from "bird flying"
    attends to source K,V differently → pose diverges while appearance transfers.
    K is NEVER injected at non-TIER_A (position-dependent) layers — any source K
    there causes RoPE spatial locking → output frozen to source pose.

    SynPS colour preservation (CVPR 2026, arXiv:2512.14423):
    ──────────────────────────────────────────────────────────
    At TIER_A layers, source K is re-encoded with a w-scaled RoPE before injection:

        cos_w = w * cos + (1 - w)   # w=0 → identity; w=1 → original
        sin_w = w * sin
        K_src_injected = apply_rotary_emb(K_src_raw, (cos_w, sin_w))

    w=0 → position-agnostic: edit Q attends to source tokens by content (colour,
           texture) rather than spatial location → colour is retrieved naturally.
    w=1 → full RoPE: identical to FreeFlux baseline (spatial lock at TIER_A).

    Adaptive w (per denoising step) via editing measurement M_t:
        M_t  = mean over TIER_A blocks of (S_img / S_txt)
        S_img = cos_sim(attn_out_img_src, attn_out_img_edit)
        S_txt = cos_sim(attn_out_txt_src, attn_out_txt_edit)
    Piecewise schedule:
        M_t > m_max (1.0): under-editing → w = 0 (agnostic, colour retrieval)
        M_t < m_min (0.9): over-editing  → w = 1 (full lock)
        else:               w = (m_max - M_t) / (m_max - m_min)
    First step uses w = 1.0 (no prior measurement).

    Parameters
    ----------
    synps             : Enable SynPS adaptive w-scaling (default True).
    m_min, m_max      : SynPS editing-measurement thresholds (default 0.9, 1.0).
    inject_layers     : K+V injection layers. Default TIER_A (13 layers).
    inject_steps_frac : Denoising step window for injection. Default all steps.
    v_blend           : Source V blend at non-TIER_A layers (default 0.0 with
                        synps=True; 0.3 recommended if synps=False as fallback).
    v_blend_steps_frac: Step window for non-TIER_A V blend.
    preserve_color    : Apply Reinhard LAB post-processing fallback.
    """

    def __init__(
        self,
        inject_layers: Optional[List[int]] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 1.0),
        v_blend: float = 0.0,
        v_blend_steps_frac: Tuple[float, float] = (0.0, 1.0),
        preserve_color: bool = False,
        synps: bool = True,
        m_min: float = 0.7,
        m_max: float = 0.95,
        static_w: Optional[float] = None,
        # Identity frequency guidance
        identity_guidance: bool = False,
        identity_strength: float = 0.3,
        identity_steps_frac: Tuple[float, float] = (0.0, 0.5),
        low_freq_cutoff: float = 0.1,
        # KV-Edit masked mode: segment subject → bg frozen, fg free to change pose
        fg_mask: Optional[Image.Image] = None,
        subject_noun: Optional[str] = None,       # ConceptAttention mask (no SAM2 needed)
        use_sam2: bool = False,
        sam2_model_id: str = "facebook/sam2-hiera-large",
        bg_dilate: int = 6,   # token-space dilation radius for soft bg composite
        # kept for backward-compat; ignored
        inject_all_single: bool = False,
        bg_steps_frac: Tuple[float, float] = (0.0, 1.0),
    ):
        self.inject_layers      = set(inject_layers if inject_layers is not None else TIER_A)
        self.inject_steps_frac  = inject_steps_frac
        self.v_blend            = v_blend
        self.v_blend_steps_frac = v_blend_steps_frac
        self.preserve_color     = preserve_color
        self.synps              = synps
        self.m_min              = m_min
        self.m_max              = m_max
        self.static_w           = static_w
        self.identity_guidance   = identity_guidance
        self.identity_strength   = identity_strength
        self.identity_steps_frac = identity_steps_frac
        self.low_freq_cutoff     = low_freq_cutoff
        self.raw_fg_mask         = fg_mask
        self.subject_noun        = subject_noun
        self.use_sam2            = use_sam2
        self.sam2_model_id       = sam2_model_id
        self.bg_dilate           = bg_dilate
        self._txt_len_single     = 512
        # SynPS per-generation state
        self.current_w: float        = static_w if static_w is not None else 1.0
        self._last_step: int         = -1
        self._m_scratch: List[float] = []
        # Side-channel state written by sampler before each inject_qkv call
        self._raw_k      = None
        self._rotary_emb = None
        # Latent dims captured in pre_generate for post_step
        self._lat_h: int = 128
        self._lat_w: int = 128
        self._lat_C: int = 16
        # KV-Edit: foreground token mask — (n_img,) bool, True = subject (editable)
        self._fg_token_mask: Optional[torch.Tensor] = None

    def pre_generate(
        self, pipe,
        max_sequence_length: int = 512,
        device: str = "cuda",
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 28,
        seed: int = 0,
        source_prompt: str = "",
        guidance_scale: float = 3.5,
        **kwargs,
    ):
        self._txt_len_single = max_sequence_length
        self.current_w  = self.static_w if self.static_w is not None else 1.0
        self._last_step = -1
        self._m_scratch = []
        self._lat_h = height // 8
        self._lat_w = width  // 8
        self._lat_C = pipe.transformer.config.in_channels // 4

        # ── KV-Edit mask setup ─────────────────────────────────────────────────
        # Priority:
        #   1. User-supplied fg_mask              (highest quality)
        #   2. ConceptAttention (subject_noun)    (semantic, no external model)
        #   3. SAM2 automatic segmentation        (geometric fallback)
        self._fg_token_mask = None
        raw_mask = self.raw_fg_mask

        if raw_mask is None and self.subject_noun:
            print(f"[NonRigid] DAAM mask for '{self.subject_noun}' "
                  f"(Q_img×K_concept, layers 10-17)…")
            raw_mask = _concept_attention_mask(
                pipe, source_prompt, self.subject_noun, seed,
                height, width, num_steps, guidance_scale, max_sequence_length, device,
            )
            if raw_mask is None:
                print("[NonRigid] ConceptAttention failed — trying SAM2 fallback.")

        if raw_mask is None and self.use_sam2:
            print("[NonRigid] SAM2 automatic segmentation…")
            src_img = _generate_source_preview(
                pipe, source_prompt, seed, height, width,
                num_steps, guidance_scale, max_sequence_length, device,
            )
            if src_img is not None:
                raw_mask = _auto_fg_mask_sam2(
                    src_img, device=device, sam2_model_id=self.sam2_model_id)
                if raw_mask is None:
                    print("[NonRigid] SAM2 found no mask — falling back to global TIER_A mode.")
            else:
                print("[NonRigid] Source preview failed — no mask.")

        if raw_mask is not None:
            # Token grid: each token = 2×2 patch of the latent (H//8 × W//8)
            th = height // 16
            tw = width  // 16
            mask_np = np.array(raw_mask.convert("L").resize((tw, th), Image.NEAREST))
            self._fg_token_mask = torch.tensor(
                mask_np > 127, dtype=torch.bool
            ).reshape(th * tw)
            n_fg  = int(self._fg_token_mask.sum())
            n_tot = th * tw
            print(f"[NonRigid] KV-Edit mask: {n_fg}/{n_tot} foreground tokens "
                  f"({100 * n_fg // n_tot}%)")

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        img_offset = txt_len if txt_len > 0 else self._txt_len_single

        # Step boundary: finalize M_t and update SynPS w.
        if step != self._last_step:
            if self.static_w is not None:
                self.current_w = self.static_w
            elif self._last_step >= 0 and self.synps and self._m_scratch:
                M = sum(self._m_scratch) / len(self._m_scratch)
                if M > self.m_max:
                    self.current_w = 0.0
                elif M < self.m_min:
                    self.current_w = 1.0
                else:
                    self.current_w = (self.m_max - M) / (self.m_max - self.m_min)
            self._m_scratch = []
            self._last_step = step

        # ── Masked mode (KV-Edit): subject segmented ─────────────────────────
        if self._fg_token_mask is not None:
            return self._masked_inject_qkv(q, k, v, layer, img_offset,
                                           _step_active(step, n_steps, self.inject_steps_frac))

        # ── Global mode (no mask) ─────────────────────────────────────────────
        inject_active = _step_active(step, n_steps, self.inject_steps_frac)
        vblend_active = (self.v_blend > 0.0
                         and _step_active(step, n_steps, self.v_blend_steps_frac))

        if not inject_active and not vblend_active:
            return q, k, v

        if layer in self.inject_layers:
            if inject_active:
                if self.synps and self._raw_k is not None and self._rotary_emb is not None:
                    k, v = self._synps_inject(k, v, img_offset)
                else:
                    k, v = _kv_full_inject(k, v, img_offset)
        elif vblend_active:
            v_src, v_edit = v.chunk(2)
            v_edit = v_edit.clone()
            v_edit[:, :, img_offset:, :] = (
                self.v_blend * v_src[:, :, img_offset:, :]
                + (1.0 - self.v_blend) * v_edit[:, :, img_offset:, :]
            )
            v = torch.cat([v_src, v_edit])

        return q, k, v

    def _masked_inject_qkv(self, q, k, v, layer, img_offset, inject_active):
        """
        KV-Edit (ICCV 2025, arXiv:2502.17363) adapted for pose editing.

        Two-zone attention injection (TIER_A only; non-TIER_A fully free):

          TIER_A layers only (13 content-similarity, low-RoPE-frequency blocks):
              Background tokens:
                  K_edit[bg] = K_src[bg],  V_edit[bg] = V_src[bg]
                  → Content anchor: at TIER_A attention is feature-driven, not
                    position-driven.  Tells the edit branch "what the bg looks
                    like" without spatially locking any token to its source grid
                    position.  Background pixels restored by post_step compositing.
              Foreground tokens (when inject_active):
                  K_edit[fg] = SynPS(K_src_raw[fg], w),  V_edit[fg] = V_src[fg]
                  → Appearance anchor: w=0 position-agnostic (max colour lock),
                    w=1 full RoPE (FreeFlux baseline).

          Non-TIER_A layers (44 position-sensitive, high-RoPE-frequency blocks):
              All tokens completely free — no injection at all.
              → Any source K at these layers would RoPE-lock tokens to their
                source positions, preventing limbs/wings from extending into
                background space.  Edit Q from "bird flying" freely restructures
                the spatial attention pattern.  post_step soft composite then
                restores far-background latents after each denoising step.
        """
        fg     = self._fg_token_mask.to(k.device)          # (n_img,) bool
        bg     = ~fg
        bg_abs = img_offset + bg.nonzero(as_tuple=True)[0]
        fg_abs = img_offset + fg.nonzero(as_tuple=True)[0]

        k = k.clone()
        v = v.clone()

        if layer in self.inject_layers:
            # TIER_A (content-similarity, low RoPE frequency):
            # ── Background: source K,V as appearance anchor ───────────────────
            # At TIER_A attention is driven by feature content, not spatial
            # position.  Injecting source K,V here tells the edit branch "what
            # the background looks like" without spatially locking it — so new
            # limbs/wings can still extend into background space at non-TIER_A.
            k[1, :, bg_abs, :] = k[0, :, bg_abs, :]
            v[1, :, bg_abs, :] = v[0, :, bg_abs, :]
            # ── Foreground: SynPS appearance anchor ───────────────────────────
            if inject_active:
                if self.synps and self._raw_k is not None and self._rotary_emb is not None:
                    w = self.current_w
                    k_src_raw = self._raw_k[0:1]
                    if w < 1.0 - 1e-6:
                        cos, sin = self._rotary_emb
                        k_src_w = apply_rotary_emb(
                            k_src_raw, (w * cos + (1.0 - w), w * sin))
                    else:
                        k_src_w = k[0:1]
                    k[1:2, :, fg_abs, :] = k_src_w[0:1, :, fg_abs, :]
                else:
                    k[1, :, fg_abs, :] = k[0, :, fg_abs, :]
                v[1, :, fg_abs, :] = v[0, :, fg_abs, :]

        # Non-TIER_A (position-sensitive, high RoPE frequency): ALL tokens free.
        # No source K,V injection here — injecting source K at position-dependent
        # layers spatially locks tokens to source positions.  Extended limbs/wings
        # that move into background space need these layers completely free so the
        # edit prompt ("flying", "jumping") can restructure spatial attention.
        # Background pixel preservation is handled by post_step soft compositing.
        return q, k, v

    def _synps_inject(self, k, v, img_offset: int):
        """
        SynPS TIER_A injection: re-encode source K with w-scaled RoPE then inject.
        Used in global (no-mask) mode.  Masked mode uses _masked_inject_qkv instead.
        """
        w = self.current_w
        k_src_raw, _            = self._raw_k.chunk(2)
        k_src_rope, k_edit_rope = k.chunk(2)
        v_src, v_edit           = v.chunk(2)

        if w < 1.0 - 1e-6:
            cos, sin = self._rotary_emb
            k_src_w = apply_rotary_emb(k_src_raw, (w * cos + (1.0 - w), w * sin))
        else:
            k_src_w = k_src_rope

        k_edit_new = k_edit_rope.clone()
        k_edit_new[:, :, img_offset:, :] = k_src_w[:, :, img_offset:, :]
        v_edit_new = v_edit.clone()
        v_edit_new[:, :, img_offset:, :] = v_src[:, :, img_offset:, :]

        return torch.cat([k_src_rope, k_edit_new]), torch.cat([v_src, v_edit_new])

    def observe_attn_out(self, out, layer, step, txt_len):
        """
        Accumulate M_t = S_img / S_txt for SynPS adaptive w.
        In masked mode, similarity is measured on foreground tokens only —
        background tokens always have identical K,V (source injected), which
        would otherwise push S_img artificially high and keep w near 0 forever.
        """
        if not self.synps or layer not in self.inject_layers:
            return

        img_offset = txt_len if txt_len > 0 else self._txt_len_single
        src = out[0]
        tgt = out[1]

        def _cos(a, b):
            af = a.detach().float().flatten()
            bf = b.detach().float().flatten()
            return (af @ bf) / (af.norm() * bf.norm() + 1e-8)

        s_txt = _cos(src[:, :txt_len, :], tgt[:, :txt_len, :]).item() if txt_len > 0 else 1.0

        if self._fg_token_mask is not None:
            fg_abs = img_offset + self._fg_token_mask.to(out.device).nonzero(as_tuple=True)[0]
            s_img = _cos(src[:, fg_abs, :], tgt[:, fg_abs, :]).item()
        else:
            s_img = _cos(src[:, img_offset:, :], tgt[:, img_offset:, :]).item()

        self._m_scratch.append(s_img / max(s_txt, 0.01))

    def post_step(self, latents, step, n_steps):
        # FFT colour anchor for foreground identity.
        if self.identity_guidance and _step_active(step, n_steps, self.identity_steps_frac):
            latents = _freq_identity_guidance(
                latents, self._lat_h, self._lat_w, self._lat_C,
                self.identity_strength, self.low_freq_cutoff,
            )

        # Background latent compositing (masked mode only).
        #
        # WHY NO HARD BINARY COMPOSITE:
        # The fg mask is computed from the SOURCE object position.  After pose
        # change the object extends into new positions that the mask labels as
        # background.  Hard-replacing those positions with source latent erases
        # the new limbs/wings and leaves a ghost of the original pose.
        #
        # Instead we use a soft composite with a DILATED and BLURRED fg mask:
        #   • Dilation (max-pool, radius bg_dilate tokens) expands the "safe fg"
        #     zone so the new pose has room to extend beyond the source boundary.
        #   • Gaussian blur (avg-pool) smooths the boundary so there is no hard
        #     seam between composited background and free foreground.
        #   • Alpha=1 → keep edit latent (fg & extended zone).
        #     Alpha=0 → composite source latent (far background, unchanged).
        #
        # Background preservation is primarily handled at the attention level in
        # _masked_inject_qkv (bg tokens always receive source K,V at all 57
        # layers); this soft composite is a supplementary anchor for far bg only.
        if self._fg_token_mask is not None:
            th = self._lat_h // 2
            tw = self._lat_w // 2
            fg_2d = (self._fg_token_mask
                     .reshape(th, tw)
                     .float()
                     .to(latents.device)
                     .unsqueeze(0).unsqueeze(0))   # (1,1,th,tw)

            # Dilate: give the edited object room to extend beyond source boundary
            dil = self.bg_dilate
            fg_dil = F.max_pool2d(fg_2d, kernel_size=2*dil+1, stride=1, padding=dil)

            # Blur: smooth the boundary to avoid seams
            fg_soft = F.avg_pool2d(fg_dil, kernel_size=5, stride=1, padding=2)

            alpha = fg_soft.reshape(th * tw, 1)    # (S, 1) — 1=keep edit, 0=composite src
            latents = latents.clone()
            # latents: (2, S, C_packed)
            latents[1] = alpha * latents[1] + (1.0 - alpha) * latents[0]

        return latents

    def post_process(self, src_img: Image.Image, edit_img: Image.Image) -> Image.Image:
        if self.preserve_color:
            return _reinhard_color_transfer(edit_img, src_img)
        return edit_img


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
            pred_iou_thresh=0.86,
            stability_score_thresh=0.88,  # lower → keeps whole-body masks
        )

        img_np = np.array(source_image.convert("RGB"))
        masks  = generator.generate(img_np)

        if not masks:
            print("[UltimateFlux] SAM2 found no masks in the source image.")
            return None

        H, W = img_np.shape[:2]
        total_px = H * W
        cx, cy = W / 2.0, H / 2.0

        def _score(m):
            bx = m['bbox'][0] + m['bbox'][2] / 2.0
            by = m['bbox'][1] + m['bbox'][3] / 2.0
            dist_norm  = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / max(W, H)
            area_frac  = m['area'] / total_px
            # Reward stability and larger area; penalise off-centre masks.
            return m['stability_score'] + 0.4 * area_frac - 0.5 * dist_norm

        best = max(masks, key=_score)
        best_seg = best['segmentation'].astype(bool)
        best_area = best_seg.sum()

        # Merge any mask that overlaps ≥20% of the best mask (union partial parts).
        combined = best_seg.copy()
        for m in masks:
            if m is best:
                continue
            seg = m['segmentation'].astype(bool)
            overlap = (seg & best_seg).sum()
            if overlap >= 0.20 * best_area:
                combined |= seg

        # Morphological closing: fill small holes so the whole body is covered.
        try:
            from scipy.ndimage import binary_fill_holes, binary_closing
            struct = np.ones((15, 15), dtype=bool)
            combined = binary_closing(combined, structure=struct)
            combined = binary_fill_holes(combined)
        except ImportError:
            pass  # scipy not installed — skip hole-filling

        fg = (combined.astype(np.uint8) * 255)
        return Image.fromarray(fg)

    except Exception as exc:
        print(f"[UltimateFlux] SAM2 segmentation failed: {exc}")
        import traceback; traceback.print_exc()
        return None


class _DAAmAttnProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 that additionally captures
    DAAM-style Q_img × K_concept attention logits at specified double-stream layers.

    Direction: Q_img × K_concept measures "how much does image patch i attend to
    concept key j."  This is the correct (non-inverted) direction for subject
    saliency: patches that belong to the concept attend most strongly to its keys.

    The previous approach (cosine sim between post-projection outputs from the same
    normal pass) was inverted because the text output at "bird" positions encodes
    "bird in context of branch/perch" — making background patches more similar to
    that contaminated text vector than the bird body itself.  Q × K logits bypass
    this contamination because they are computed before any inter-token mixing.
    """

    def __init__(
        self,
        concept_layers: List[int],
        concept_token_idx: List[int],
        n_img: int,
        n_layers_total: int = 57,
    ):
        self.concept_layers    = set(concept_layers)
        self.concept_token_idx = concept_token_idx
        self.n_img             = n_img
        self.n_layers_total    = n_layers_total
        self._cur_layer        = 0
        self._accum            = torch.zeros(n_img, dtype=torch.float32)
        self._n_cap            = 0

    @property
    def saliency(self) -> torch.Tensor:
        return self._accum / max(1, self._n_cap)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        # Explicit named params so inspect.signature passes them without warnings:
        step_index=None,
        index_block=None,
        mm_copy_blocks=None,
        single_copy_blocks=None,
        attention_store=None,
    ):
        is_double = encoder_hidden_states is not None
        cur_layer = self._cur_layer
        B         = hidden_states.shape[0]
        inner_dim = attn.to_k.out_features if hasattr(attn.to_k, 'out_features') else attn.inner_dim
        head_dim  = inner_dim // attn.heads

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

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

        # ── DAAM saliency capture ─────────────────────────────────────────────
        if is_double and cur_layer in self.concept_layers:
            valid_c     = [ci for ci in self.concept_token_idx if ci < txt_len]
            n_img_local = q.shape[2] - txt_len
            if valid_c and n_img_local == self.n_img:
                q_img  = q[:, :, txt_len:, :].float()   # (B, H, n_img, D)
                k_con  = k[:, :, valid_c,  :].float()   # (B, H, n_con, D)
                # logit: how strongly does image patch i attend to concept token j
                logits = torch.einsum('bhid,bhjd->bhij', q_img, k_con) * (head_dim ** -0.5)
                # average over heads and concept tokens → (B, n_img)
                sal = logits.mean(1).mean(-1)
                # use the conditional branch (last item in batch when CFG doubles it)
                self._accum.add_(sal[-1].detach().cpu())
                self._n_cap += 1
        # ─────────────────────────────────────────────────────────────────────

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            self._cur_layer = (self._cur_layer + 1) % self.n_layers_total
            return out, enc_out

        # Single-stream blocks (FluxSingleTransformerBlock) use Attention(pre_only=True)
        # which has no to_out layer — the block applies proj_out externally after
        # concatenating attn_output with mlp_hidden_states.  Return directly.
        self._cur_layer = (self._cur_layer + 1) % self.n_layers_total
        return out


def _concept_attention_mask(
    pipe,
    source_prompt: str,
    subject_noun: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    device: str,
    concept_layers: Optional[List[int]] = None,
    top_k_frac: float = 0.35,
    mask_steps: int = 20,
) -> Optional[Image.Image]:
    """
    DAAM-style subject mask from FLUX cross-attention (arXiv:2502.04320).

    At double-stream layers 10-17, captures Q_img × K_concept pre-softmax logits:
    "how much does image patch i attend to concept key j?"  Averaged over heads,
    concept tokens, and denoising steps, then thresholded at top_k_frac percentile
    to produce a binary token mask.

    Fixes the previous output-projection cosine-similarity approach which was
    contaminated by full-prompt context and produced inverted (background-as-fg)
    masks, causing the copy-paste double-image artifact.
    """
    if concept_layers is None:
        concept_layers = list(range(10, 18))

    tok = getattr(pipe, 'tokenizer_2', None) or getattr(pipe, 'tokenizer', None)
    if tok is None:
        print("[ConceptMask] No T5 tokenizer found.")
        return None
    try:
        prompt_ids = tok(source_prompt, add_special_tokens=False).input_ids
        noun_ids   = tok(subject_noun,  add_special_tokens=False).input_ids
    except Exception as exc:
        print(f"[ConceptMask] Tokenizer error: {exc}")
        return None

    concept_idx: List[int] = []
    for i in range(len(prompt_ids) - len(noun_ids) + 1):
        if prompt_ids[i:i + len(noun_ids)] == noun_ids:
            concept_idx = list(range(i, i + len(noun_ids)))
            break
    if not concept_idx:
        for wt in noun_ids:
            concept_idx += [j for j, pt in enumerate(prompt_ids) if pt == wt]
        concept_idx = sorted(set(concept_idx))
    if not concept_idx:
        print(f"[ConceptMask] '{subject_noun}' not found in T5 token sequence.")
        return None

    print(f"[ConceptMask] concept_idx={concept_idx} (tokens for '{subject_noun}')")

    th    = height // 16
    tw    = width  // 16
    n_img = th * tw

    proc = _DAAmAttnProcessor(concept_layers, concept_idx, n_img)

    # Install our processor; restore afterwards (orig_procs is a dict of per-block procs)
    orig_procs = pipe.transformer.attn_processors
    pipe.transformer.set_attn_processor(proc)

    try:
        exec_device = getattr(pipe, '_execution_device', device)
        generator   = torch.Generator(device=exec_device).manual_seed(seed)
        with torch.no_grad():
            pipe(
                prompt=source_prompt,
                num_inference_steps=mask_steps,
                guidance_scale=1.0,          # no CFG → B=1 per step → 2× faster
                height=height,
                width=width,
                generator=generator,
                max_sequence_length=max_sequence_length,
                output_type="latent",
            )
    except Exception as exc:
        print(f"[ConceptMask] Forward pass failed: {exc}")
        return None
    finally:
        pipe.transformer.set_attn_processor(orig_procs)

    if proc._n_cap == 0:
        print("[ConceptMask] No captures — concept_idx may exceed txt_len.")
        return None

    saliency = proc.saliency    # (n_img,) mean over steps × layers × concepts

    k_top    = max(1, int(top_k_frac * n_img))
    topk_val = float(saliency.topk(k_top).values[-1])
    binary   = (saliency >= topk_val).numpy().reshape(th, tw).astype(np.uint8) * 255
    mask_pil = Image.fromarray(binary, mode="L").resize((width, height), Image.NEAREST)

    n_fg = int((saliency >= topk_val).sum())
    print(f"[ConceptMask] '{subject_noun}': {n_fg}/{n_img} fg tokens "
          f"({100 * n_fg // n_img}%), captures={proc._n_cap}")
    return mask_pil


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
        use_sam2: bool = False,
        sam2_model_id: str = "facebook/sam2-hiera-large",
    ):
        self.top_k_frac      = top_k_frac
        self.qk_frac         = qk_frac
        self.v_frac          = v_frac
        self.color_word      = color_word
        self.chunk_size      = chunk_size
        self.mask_build_step = mask_build_step
        self.reweight_scale  = reweight_scale
        self.ds_key_inject   = ds_key_inject
        self.use_sam2        = use_sam2
        self.sam2_model_id   = sam2_model_id
        self._txt_len        = 512
        self._color_tok_ids: List[int] = []
        self._mask: Optional[torch.Tensor] = None       # (n_img,) preserve mask
        self._mask_built     = False
        self._edit_score_map: Optional[torch.Tensor] = None  # (n_img,) CPU — for SAM2

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
        self._edit_score_map = None
        self._color_tok_ids = []
        if self.color_word and edit_prompt:
            self._color_tok_ids = _get_t5_token_indices(pipe, edit_prompt, self.color_word)
            print(f"[ColorCtrl] '{self.color_word}' → T5 indices: {self._color_tok_ids}")

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        # Double-stream K+V injection: locks both spatial layout (K) and output
        # values (V) so the representation entering single-stream is source-like.
        # K-only was insufficient: without V, 38 free single-stream blocks generate
        # a completely different person/car.  V injection anchors identity strongly;
        # colour change is then driven by re-weighting in single-stream via V_txt.
        if self.ds_key_inject and txt_len > 0:
            img_offset = txt_len
            k_src, k_tgt = k.chunk(2)
            v_src, v_tgt = v.chunk(2)
            k_tgt_new = k_tgt.clone()
            v_tgt_new = v_tgt.clone()
            k_tgt_new[:, :, img_offset:, :] = k_src[:, :, img_offset:, :]
            v_tgt_new[:, :, img_offset:, :] = v_src[:, :, img_offset:, :]
            k = torch.cat([k_src, k_tgt_new])
            v = torch.cat([v_src, v_tgt_new])
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
        Structure-colour separated injection for single-stream blocks.

        Algorithm
        ---------
        1. Mask: image patches attending most to the colour word → editing region.
           Derived from target branch img→text attention at mask_build_step.
        2. ALL image tokens  → source K   (structure lock everywhere)
           Every image patch attends to the SAME spatial positions as in source,
           preserving geometry for both the editing region (hair/car shape) and
           non-editing region (face, background).
        3. NON-editing tokens → source V  (identity lock: face, bg = source)
           EDITING tokens     → target V  (colour free: hair/car conditioned on
           edit text, e.g. "blonde" / "blue")
        4. Standard SDPA — no manual attention loops needed.

        Parameters (reused from __init__)
        ----------------------------------
        qk_frac : step fraction for K injection (default 1.0 = all steps).
        v_frac  : step fraction for V masking   (default 1.0 = all steps).
        top_k_frac      : fraction of image tokens in the editing mask.
        mask_build_step : first step at which the mask is extracted.
        color_word      : word in edit_prompt whose T5 tokens focus the mask.
        ds_key_inject   : K+V injection in double-stream blocks for coarse anchor.
        """
        # ── Double-stream: handled by inject_qkv ─────────────────────────────
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
        head_dim = q.shape[-1]
        scale    = head_dim ** -0.5

        do_struct = step < int(self.qk_frac * n_steps)   # K injection active
        do_colour = step < int(self.v_frac  * n_steps)   # V masking active

        # ── 1. Mask extraction ───────────────────────────────────────────────
        # Wait until mask_build_step so the model has formed spatial structure.
        # Use the target branch's img→text attention to find which image patches
        # attend most strongly to the colour-word token(s).
        if not self._mask_built and step >= self.mask_build_step and layer >= N_DOUBLE:
            q_tgt    = q[B // 2:]
            k_tgt_tx = k[B // 2:, :, :text_len, :]
            # img→text raw (pre-softmax) scores averaged over heads
            v2t      = torch.matmul(
                q_tgt[:, :, text_len:, :].float(),
                k_tgt_tx.float().transpose(-2, -1),
            ) * scale                                      # (1, H, n_img, n_txt)
            v2t_mean = v2t.mean(dim=(0, 1))               # (n_img, n_txt)

            if self._color_tok_ids:
                ids        = [i for i in self._color_tok_ids if i < text_len]
                edit_score = (v2t_mean[:, ids].mean(dim=-1)
                              if ids else v2t_mean.mean(dim=-1))
            else:
                edit_score = v2t_mean.mean(dim=-1)        # (n_img,)

            if not self.use_sam2:
                # Attention-based topk mask (default, no extra dependencies)
                k_top        = max(1, int(self.top_k_frac * n_img))
                edit_idx     = edit_score.topk(k_top).indices
                preserve_mask            = torch.ones(n_img, device=q.device,
                                                      dtype=torch.float32)
                preserve_mask[edit_idx] = 0.0
                self._mask               = preserve_mask
                self._mask_built         = True
                print(f"[ColorCtrl] attention mask built at layer={layer} step={step}: "
                      f"{k_top}/{n_img} editing tokens "
                      f"({'colour-word focused' if self._color_tok_ids else 'all-text'})")
            else:
                # SAM2 path: store scores on CPU; the step-end callback will decode
                # the latent and run SAM2 with the attention-peak as a point prompt.
                self._edit_score_map = edit_score.detach().cpu()
                print(f"[ColorCtrl] attention scores stored at step={step} layer={layer} "
                      f"— SAM2 will segment in step-end callback")

        # Need the mask for any injection
        if self._mask is None:
            return None
        if not do_struct and not do_colour:
            return None

        # ── 2. Structure: ALL image tokens → source K ────────────────────────
        # Replaces the target's image K with source's, so every patch attends to
        # the same spatial positions as in the source image.  Preserves geometry
        # of the editing object (hair shape, car silhouette) AND the face/bg.
        k_src, k_tgt = k.chunk(2)
        v_src, v_tgt = v.chunk(2)
        k_tgt_new    = k_tgt.clone()
        v_tgt_new    = v_tgt.clone()

        if do_struct:
            k_tgt_new[:, :, text_len:, :] = k_src[:, :, text_len:, :]

        # ── 3. Colour: split editing region ─────────────────────────────────
        # NON-editing (face, bg)  → source V : these regions are identity-locked.
        # EDITING (hair, car)     → target V : free to adopt the new colour from
        #                                       the edit text ("blonde", "blue").
        if do_colour:
            non_edit = self._mask.to(device=v.device,
                                     dtype=v.dtype).view(1, 1, -1, 1)  # 1=non-edit
            v_tgt_new[:, :, text_len:, :] = (
                v_src[:, :, text_len:, :] * non_edit               # non-edit: source
                + v_tgt[:, :, text_len:, :] * (1.0 - non_edit)    # editing : target
            )

        k_new = torch.cat([k_src, k_tgt_new])
        v_new = torch.cat([v_src, v_tgt_new])

        # ── 4. Standard SDPA with modified K, V ─────────────────────────────
        return F.scaled_dot_product_attention(
            q, k_new, v_new, dropout_p=0.0, is_causal=False
        )


# ─────────────────────────── Task 5c: Kontext-style colour editing ───────────

class KontextColorPolicy(BasePolicy):
    """
    Kontext-style structure-preserving colour editing for FLUX.1-dev.

    Two-tier injection strategy:

    1. Double-stream blocks (0-18) — Full Kontext SAC:
       Edit image-token Q AND K ← source Q and K.
       Text Q/K stay from the edit branch → edit prompt conditions V freely.

    2. Single-stream blocks (19-56) — graduated injection:
       • All steps: image K ← source K  (spatial position lock — always on).
       • First ss_q_steps_frac of steps: image Q ← source Q too.
         Q lock in early steps prevents edit-conditioned queries from drifting
         the structure while identity is still forming.
       • Remaining steps: K-only.  Edit Q is free → edit text can steer V toward
         the new colour without fighting the source's structural queries.

    This gives identity lock during structure formation and colour freedom
    during the appearance-refinement phase where colour actually settles.

    Optional PFB (svd_alpha > 0):
        Forward hook at selected double-stream blocks.
        h_edit ← Φ(h_src, α) + (h_edit − Φ(h_edit, α))

    Parameters
    ----------
    qk_layers         : double-stream layers for Q+K injection (default: 0-18).
    k_only_layers     : single-stream layers for K (+ early-Q) injection (default: 19-56).
    qk_steps_frac     : step fraction for double-stream Q+K.  Default all steps.
    k_only_steps_frac : step fraction for single-stream K injection.  Default all.
    ss_q_steps_frac   : step fraction for single-stream Q injection.  Default (0, 0.5).
                        Increase end toward 1.0 for tighter identity; decrease for
                        stronger colour.
    svd_alpha         : PFB reweighting factor.  0 = disabled (default).
    svd_layers        : blocks for PFB hook.  Default [1].
    svd_steps_frac    : step fraction for PFB.  Default first 25 %.
    """

    def __init__(
        self,
        qk_layers: Optional[List[int]] = None,
        k_only_layers: Optional[List[int]] = None,
        qk_steps_frac: Tuple[float, float] = (0.0, 1.0),
        k_only_steps_frac: Tuple[float, float] = (0.0, 1.0),
        ss_q_steps_frac: Tuple[float, float] = (0.0, 0.5),
        svd_alpha: float = 0.0,
        svd_layers: Optional[List[int]] = None,
        svd_steps_frac: Tuple[float, float] = (0.0, 0.25),
    ):
        self._qk_layers        = set(qk_layers if qk_layers is not None
                                     else range(N_DOUBLE))
        self._k_only_layers    = set(k_only_layers if k_only_layers is not None
                                     else range(N_DOUBLE, N_LAYERS))
        self.qk_steps_frac     = qk_steps_frac
        self.k_only_steps_frac = k_only_steps_frac
        self.ss_q_steps_frac   = ss_q_steps_frac
        self.svd_alpha         = svd_alpha
        self._svd_layers       = set(svd_layers if svd_layers is not None else [1])
        self.svd_steps_frac    = svd_steps_frac
        self._n_steps          = 28
        self._txt_len          = 512       # updated in pre_generate
        self._layer_step: Dict[int, int] = {}

    def pre_generate(self, pipe, num_steps: int = 28,
                     max_sequence_length: int = 512, **kwargs):
        self._n_steps = num_steps
        self._txt_len = max_sequence_length
        self._layer_step = {lid: 0 for lid in self._svd_layers}

    def inject_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        step: int,
        n_steps: int,
        txt_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q.shape[0] < 2:
            return q, k, v

        is_double = txt_len > 0

        if is_double:
            # ── Double-stream: Q+K injection (Kontext SAC) ──────────────────
            if layer not in self._qk_layers:
                return q, k, v
            if not _step_active(step, n_steps, self.qk_steps_frac):
                return q, k, v
            img_offset = txt_len
            q_src, q_edit = q.chunk(2)
            k_src, k_edit = k.chunk(2)
            q_edit = q_edit.clone()
            k_edit = k_edit.clone()
            q_edit[:, :, img_offset:, :] = q_src[:, :, img_offset:, :]
            k_edit[:, :, img_offset:, :] = k_src[:, :, img_offset:, :]
            return torch.cat([q_src, q_edit]), torch.cat([k_src, k_edit]), v

        else:
            # ── Single-stream: graduated injection ──────────────────────────
            # K injection active for the full k_only_steps_frac window (always
            # locks spatial positions).  Q injection active only for the shorter
            # ss_q_steps_frac window (locks structural queries in early steps,
            # then releases so edit text can strengthen colour via V in late steps).
            if layer not in self._k_only_layers:
                return q, k, v
            if not _step_active(step, n_steps, self.k_only_steps_frac):
                return q, k, v
            img_offset = self._txt_len
            k_src, k_edit = k.chunk(2)
            k_edit = k_edit.clone()
            k_edit[:, :, img_offset:, :] = k_src[:, :, img_offset:, :]
            new_k = torch.cat([k_src, k_edit])

            if _step_active(step, n_steps, self.ss_q_steps_frac):
                q_src, q_edit = q.chunk(2)
                q_edit = q_edit.clone()
                q_edit[:, :, img_offset:, :] = q_src[:, :, img_offset:, :]
                return torch.cat([q_src, q_edit]), new_k, v

            return q, new_k, v

    def get_block_hooks(self) -> Dict[int, Callable]:
        if self.svd_alpha <= 0.0:
            return {}

        alpha       = self.svd_alpha
        svd_frac    = self.svd_steps_frac
        n_steps_ref = [self._n_steps]
        layer_step  = self._layer_step

        def make_hook(lid: int) -> Callable:
            def _hook(module, inp, out):
                cur = layer_step.get(lid, 0)
                layer_step[lid] = cur + 1
                if not _step_active(cur, n_steps_ref[0], svd_frac):
                    return out
                is_tuple = isinstance(out, tuple)
                has_two  = is_tuple and len(out) >= 2
                h = out[1] if has_two else (out[0] if is_tuple else out)
                if h.shape[0] < 2:
                    return out
                h_src      = h[0:1]
                h_edit     = h[1:2]
                h_edit_new = _apply_pfb(h_edit, h_src, alpha)
                h_out      = torch.cat([h_src, h_edit_new], dim=0)
                if has_two:
                    return (out[0], h_out)
                return (h_out,) + out[1:] if is_tuple else h_out
            return _hook

        return {lid: make_hook(lid) for lid in self._svd_layers}


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
