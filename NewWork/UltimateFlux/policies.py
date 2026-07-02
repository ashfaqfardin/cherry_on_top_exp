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

from .sampler import BasePolicy, TIER_A, TIER_B, N_DOUBLE, N_SINGLE, N_LAYERS


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


# ────────────────── Phase-1 K,V capture processor ────────────────────────────

class _KVCaptureProcessor:
    """
    Lightweight attention processor used exclusively during Phase 1 of
    ObjectAdditionPolicy.  Runs standard FLUX attention and saves the image-token
    slice of K and V at each specified layer for every denoising step.

    Saved: captured_kv[layer][step] = (k_img, v_img)
           k_img / v_img shape: (B, H, L_img, D) on CPU.
    """

    def __init__(
        self,
        capture_layers: List[int],
        total_layers: int = N_LAYERS,
        total_steps: int = 4,
        txt_len_single: int = 256,
    ):
        self.capture_layers = set(capture_layers)
        self.total_layers   = total_layers
        self.total_steps    = total_steps
        self.txt_len_single = txt_len_single   # text prefix length in single-stream blocks
        self.cur_step       = 0
        self.cur_att_layer  = 0
        self.captured_kv: Dict[int, List] = {l: [] for l in capture_layers}

    def _tick(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.total_layers:
            self.cur_att_layer = 0
            self.cur_step = (self.cur_step + 1) % self.total_steps

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        layer     = self.cur_att_layer
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

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # Save only image-token slice to avoid text-length mismatch between phases.
        # For double-stream blocks txt_len comes from encoder_hidden_states (dynamic).
        # For single-stream blocks text+image are concatenated in hidden_states; use
        # the static txt_len_single which must match the max_sequence_length passed
        # to pipe() in both Phase 1 and Phase 2.
        if layer in self.capture_layers:
            img_start = txt_len if is_double else self.txt_len_single
            self.captured_kv[layer].append((
                k[:, :, img_start:, :].detach().cpu(),
                v[:, :, img_start:, :].detach().cpu(),
            ))

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)
        out = out.to(q.dtype)

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


# ─────────────────────────── Task 2: Object addition ─────────────────────────

class ObjectAdditionPolicy(BasePolicy):
    """
    Two-phase object addition using FreeFlux's layout-aware K,V capture.

    Phase 1  (runs inside pre_generate):
        A full source-only denoising pass with _KVCaptureProcessor.
        K,V are captured at FreeFlux's position-dependent layers
        `{1,2,4,26,30,54,55}` (combined indices) at every step.
        These layers encode spatial/positional layout information.

    Phase 2  (the main generate_dual_branch call):
        B=2 denoising with [source_prompt, edit_prompt].
        Phase 1 K,V are injected into the edit branch (index 1) at the same
        hotspot layers, OUTSIDE the placement mask so the new object can emerge
        freely inside the mask while background context is frozen.

    placement_mask: binary PIL Image — 1 (white) = region where the new object
                    should appear.  If None, Phase 1 K,V are injected globally.
    """

    # FreeFlux dev position-dependent (layout-hotspot) combined layer indices.
    # Source: run_add_object.py _PROCESSOR_ARGS["layer_idx"]
    HOTSPOT_LAYERS = [1, 2, 4, 26, 30, 54, 55]

    def __init__(
        self,
        placement_mask: Optional[Image.Image] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 0.75),
    ):
        self.raw_mask           = placement_mask
        self.inject_steps_frac  = inject_steps_frac
        self._captured_kv: Dict[int, List] = {}
        self._token_mask: Optional[torch.Tensor] = None
        self._txt_len_single: int = 512   # set in pre_generate from max_sequence_length
        self._phase = 1

    def pre_generate(
        self,
        pipe,
        device: str = "cuda",
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 28,
        seed: int = 0,
        source_prompt: str = "",
        max_sequence_length: int = 512,
        guidance_scale: float = 3.5,
        **kwargs,
    ):
        """Phase 1: capture layout K,V with source-only denoising."""
        self._txt_len_single = max_sequence_length   # image tokens start at this offset in SS blocks

        if self.raw_mask is not None:
            self._token_mask = _image_mask_to_token_mask(self.raw_mask, height, width, device)

        capture_proc = _KVCaptureProcessor(
            capture_layers=self.HOTSPOT_LAYERS,
            total_layers=N_LAYERS,
            total_steps=num_steps,
            txt_len_single=max_sequence_length,   # single-stream text prefix length
        )
        pipe.transformer.set_attn_processor(capture_proc)

        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[UltimateFlux] ObjectAdditionPolicy Phase 1: capturing layout K,V …")
        _ = pipe(
            prompt=source_prompt,
            generator=generator,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,         # must match Phase 2 for consistent K,V
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,   # must match Phase 2
            output_type="pil",
        )

        self._captured_kv = capture_proc.captured_kv
        captured_counts   = {l: len(v) for l, v in self._captured_kv.items()}
        print(f"[UltimateFlux] Phase 1 done. Captured steps per layer: {captured_counts}")
        self._phase = 2

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if self._phase != 2:
            return q, k, v
        if layer not in self.HOTSPOT_LAYERS:
            return q, k, v
        if not _step_active(step, n_steps, self.inject_steps_frac):
            return q, k, v

        captured = self._captured_kv.get(layer, [])
        if step >= len(captured):
            return q, k, v

        k_p1_img, v_p1_img = captured[step]           # (1, H, L_img, D) on CPU
        k_p1_img = k_p1_img.to(k.device, dtype=k.dtype)
        v_p1_img = v_p1_img.to(v.device, dtype=v.dtype)
        # Guard against an unexpected missing batch dim
        if k_p1_img.dim() == 3:
            k_p1_img = k_p1_img.unsqueeze(0)
        if v_p1_img.dim() == 3:
            v_p1_img = v_p1_img.unsqueeze(0)

        # Image-token offset in the combined sequence:
        #   double-stream (layer < 19): txt_len is passed non-zero from sampler
        #   single-stream (layer >= 19): sampler passes txt_len=0 because there is
        #     no separate encoder_hidden_states, but the hidden_states still carry
        #     a text prefix of length _txt_len_single before the image tokens.
        img_offset = txt_len if layer < N_DOUBLE else self._txt_len_single

        # B=2: index 0 = source branch, index 1 = edit branch
        k_src, k_edit = k.chunk(2)
        v_src, v_edit = v.chunk(2)
        k_edit = k_edit.clone()
        v_edit = v_edit.clone()

        if self._token_mask is not None:
            # Outside mask (1 - mask): inject Phase 1 K,V to freeze context
            # Inside mask (mask):      keep edit K,V so new object can emerge
            outside = (1.0 - self._token_mask.squeeze()).to(k.device)  # (L_img,)
            outside4 = outside.view(1, 1, -1, 1)
            inside4  = (1.0 - outside).view(1, 1, -1, 1)
            k_edit[:, :, img_offset:, :] = k_p1_img * outside4 + k_edit[:, :, img_offset:, :] * inside4
            v_edit[:, :, img_offset:, :] = v_p1_img * outside4 + v_edit[:, :, img_offset:, :] * inside4
        else:
            # No mask: inject Phase 1 K,V into image-token positions only
            k_edit[:, :, img_offset:, :] = k_p1_img
            v_edit[:, :, img_offset:, :] = v_p1_img

        k = torch.cat([k_src, k_edit])
        v = torch.cat([v_src, v_edit])
        return q, k, v


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
        inject_steps_frac: Tuple[float, float] = (0.0, 0.8),
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
    TIER_A layers: inject source image-token K,V globally (freeze appearance/context).
    All other layers: inject outside the replacement mask only (background preservation).

    mask: binary PIL Image — 1 (white) = pixels where the object IS being replaced.
    """

    def __init__(
        self,
        mask: Optional[Image.Image] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 0.9),
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

        if layer in TIER_A:
            k, v = _kv_full_inject(k, v, img_offset)
        elif self._token_mask is not None:
            # Outside-mask injection: source K,V where mask=0, original where mask=1
            outside = (1.0 - self._token_mask.squeeze(0).squeeze(0))  # (L_img,)
            k, v = _masked_kv_inject(k, v, outside, img_offset)

        return q, k, v


# ─────────────────────────── Task 4: Background replacement ──────────────────

class BackgroundReplacePolicy(BasePolicy):
    """
    Value-only injection at all 57 layers inside the foreground mask.
    V from source flows into edit branch within the fg region → preserves subject.
    Background tokens (mask=0) are unrestricted → freely regenerated.

    fg_mask: binary PIL Image — 1 (white) = foreground object to preserve.
    """

    def __init__(self, fg_mask: Optional[Image.Image] = None):
        self.raw_mask = fg_mask
        self._token_mask: Optional[torch.Tensor] = None
        self._txt_len_single = 512  # updated in pre_generate

    def pre_generate(self, pipe, device="cuda", height=1024, width=1024,
                     max_sequence_length=512, **kwargs):
        if self.raw_mask is not None:
            self._token_mask = _image_mask_to_token_mask(self.raw_mask, height, width, device)
        self._txt_len_single = max_sequence_length

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if self._token_mask is None:
            return q, k, v

        # txt_len > 0 for double-stream; use _txt_len_single for single-stream
        img_offset = txt_len if txt_len > 0 else self._txt_len_single

        v_src, v_edit = v.chunk(2)
        v_edit = v_edit.clone()

        fg = self._token_mask.squeeze(0).squeeze(0).to(v.device)  # (L_img,)
        fg4 = fg.view(1, 1, -1, 1)
        v_edit[:, :, img_offset:, :] = (
            v_src[:, :, img_offset:, :] * fg4 + v_edit[:, :, img_offset:, :] * (1 - fg4)
        )
        v = torch.cat([v_src, v_edit])
        return q, k, v


# ─────────────────────────── Task 5: Fine-grained attribute ──────────────────

class FineGrainedAttrPolicy(BasePolicy):
    """
    Orthogonal-projection edit on image-token keys at TIER_A layers.

    Adapts FluxSpace §4: the edit vector (k_edit − k_src) for image tokens is
    projected away from the content direction k_src (Gram-Schmidt), isolating
    the attribute direction while leaving identity intact. Text tokens are always
    left untouched so each branch preserves its own prompt conditioning.
    Norm is preserved to avoid contrast collapse.

    For shape-linked edits (e.g. breed change) pass inject_layers=TIER_B.
    """

    def __init__(
        self,
        edit_scale: float = 5.0,
        inject_layers: Optional[List[int]] = None,
        inject_steps_frac: Tuple[float, float] = (0.0, 1.0),
    ):
        self.edit_scale = edit_scale
        self.inject_layers = inject_layers if inject_layers is not None else TIER_A
        self.inject_steps_frac = inject_steps_frac
        self._txt_len_single = 512  # updated in pre_generate

    def pre_generate(self, pipe, max_sequence_length: int = 512, **kwargs):
        self._txt_len_single = max_sequence_length

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if layer not in self.inject_layers:
            return q, k, v
        if not _step_active(step, n_steps, self.inject_steps_frac):
            return q, k, v

        img_offset = txt_len if txt_len > 0 else self._txt_len_single

        k_src, k_edit = k.chunk(2)

        # Work on image tokens only (positions img_offset:)
        k_src_img  = k_src[:, :, img_offset:, :]
        k_edit_img = k_edit[:, :, img_offset:, :]

        # Gram-Schmidt: project edit direction away from content direction
        edit_dir  = k_edit_img - k_src_img
        dot       = (edit_dir * k_src_img).sum(dim=-1, keepdim=True)
        norm_sq   = (k_src_img * k_src_img).sum(dim=-1, keepdim=True) + 1e-8
        orth_dir  = edit_dir - (dot / norm_sq) * k_src_img

        # Norm-preserving application
        orig_norm = torch.norm(k_edit_img, dim=-1, keepdim=True) + 1e-8
        k_new_img = k_src_img + self.edit_scale * orth_dir
        k_new_img = k_new_img / (torch.norm(k_new_img, dim=-1, keepdim=True) + 1e-8) * orig_norm

        # Rebuild: text tokens unchanged, image tokens replaced
        k_new = k_edit.clone()
        k_new[:, :, img_offset:, :] = k_new_img
        k = torch.cat([k_src, k_new])
        return q, k, v


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
        pfb_steps_frac: Tuple[float, float] = (0.0, 1.0),
    ):
        self.style_image = style_image
        self.alpha = alpha
        self.sac_steps_frac = sac_steps_frac
        self.pfb_steps_frac = pfb_steps_frac
        self._h_sty: Optional[torch.Tensor] = None
        self._step_counter = [0]     # mutable ref shared with closure

    def pre_generate(self, pipe, device="cuda", height=1024, width=1024,
                     num_steps=28, seed=0, style_image=None, **kwargs):
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
        h_sty         = self._h_sty
        alpha         = self.alpha
        pfb_frac      = self.pfb_steps_frac
        step_counter  = self._step_counter

        def _pfb_hook(module, inp, out):
            # FluxTransformerBlock returns (encoder_hidden_states, hidden_states).
            # out[0] = text hidden states (leave untouched).
            # out[1] = image hidden states — apply PFB here.
            is_tuple = isinstance(out, tuple)
            has_two  = is_tuple and len(out) >= 2

            h = out[1] if has_two else (out[0] if is_tuple else out)

            if h_sty is None or h.shape[0] < 2:
                return out

            h_edit  = h[1:2]                                 # (1, L_img, C)
            h_style = h_sty.to(h.device, dtype=h.dtype)
            h_new   = _apply_pfb(h_edit, h_style, alpha)
            h_out   = torch.cat([h[0:1], h_new], dim=0)

            step_counter[0] += 1

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
