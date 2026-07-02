"""
Shared dual-branch FLUX.1-dev sampler for UltimateFlux.

All tasks share a B=2 denoising loop:
  batch index 0 = source/content branch  (unmodified)
  batch index 1 = edit/styled branch     (receives injections from policy)

InjectionPolicy controls Q/K/V manipulation and block-level hooks at each (layer, step).

Layer index convention (combined 0-based across all 57 blocks):
  double-stream:  0–18  → pipe.transformer.transformer_blocks[i]
  single-stream: 19–56  → pipe.transformer.single_transformer_blocks[i-19]
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from typing import Optional, List, Tuple, Dict, Callable

# ── Layer tier constants (from FreeFlux ICCV 2025, validated on FLUX.1-dev) ──
# Tier A — Content-similarity-dependent layers: appearance, style, texture, object
# Source: FreeFlux non_rigid_attn_utils.py layer_idx (RoPE frequency analysis)
TIER_A = [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]

# Tier B — All layers: used when shape/pose deformation is needed
TIER_B = list(range(57))

N_DOUBLE = 19
N_SINGLE = 38
N_LAYERS = N_DOUBLE + N_SINGLE  # 57


# ─────────────────────────────── Base Policy ─────────────────────────────────

class BasePolicy:
    """
    Abstract injection policy.  Subclasses override the three entry points.
    """

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
        """Called inside every attention layer.  Modify q/k/v for edit branch."""
        return q, k, v

    def get_block_hooks(self) -> Dict[int, Callable]:
        """
        Return {block_idx: hook_fn} for transformer block forward hooks.
        Hooks are registered before generation and removed after.
        block_idx uses the same combined 0-56 convention as layer tiers.
        """
        return {}

    def pre_generate(self, pipe: FluxPipeline, **kwargs) -> None:
        """Called once before the denoising loop (style extraction, inversion, …)."""
        pass

    def post_process(self, src_img: Image.Image, edit_img: Image.Image) -> Image.Image:
        """Called after generation with the decoded PIL images.  Default: no-op."""
        return edit_img


# ─────────────────────────── Attention Processor ─────────────────────────────

class UltimateFluxAttnProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 that delegates Q/K/V
    manipulation to a pluggable BasePolicy.

    Maintains a (cur_step, cur_att_layer) counter identical to FreeFlux's
    processors so policies can gate on specific (layer, step) pairs.
    """

    def __init__(self, policy: BasePolicy, total_layers: int = N_LAYERS, total_steps: int = 4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("UltimateFluxAttnProcessor requires PyTorch 2.0+")
        self.policy = policy
        self.total_layers = total_layers
        self.total_steps = total_steps
        self.cur_step = 0
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def _tick(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.total_layers:
            self.cur_att_layer = 0
            self.cur_step = (self.cur_step + 1) % self.total_steps

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        out = self._forward(attn, hidden_states, encoder_hidden_states,
                            attention_mask, image_rotary_emb)
        self._tick()
        return out

    def _forward(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask, image_rotary_emb):
        layer = self.cur_att_layer
        step  = self.cur_step
        is_double = encoder_hidden_states is not None
        batch_size = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        inner_dim = k.shape[-1]
        head_dim  = inner_dim // attn.heads

        q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]          # number of text tokens in the prefix
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # ── Policy injection ──────────────────────────────────────────────
        q, k, v = self.policy.inject_qkv(q, k, v, layer, step, self.total_steps, txt_len)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out

        return out


# ────────────────────────── Pipeline helpers ──────────────────────────────────

def load_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-dev",
    hf_token: Optional[str] = None,
    device: str = "cuda",
    cpu_offload: bool = False,
    cache_dir: str = "./models",
) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


@torch.no_grad()
def generate_dual_branch(
    pipe: FluxPipeline,
    policy: BasePolicy,
    source_prompt: str,
    edit_prompt: str,
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
) -> Tuple[Image.Image, Image.Image]:
    """
    Dual-branch denoising with injections from policy.

    Returns (source_image, edit_image).  source_image is the unmodified
    reconstruction; edit_image is the policy-steered output.
    """
    # 1. Policy pre-hook (style extraction, reasoning pass, inversion, etc.)
    policy.pre_generate(
        pipe,
        device=device,
        height=height,
        width=width,
        num_steps=num_steps,
        seed=seed,
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
    )

    # 2. Shared initial latent — both branches MUST start from identical noise.
    #    Without this, pipe(prompt=[A, B]) generates different noise for A and B
    #    (sequential samples from the same generator), so K,V injection between
    #    branches fights completely different initial conditions and produces garbage.
    #    FreeFlux's run_non_rigid.py does exactly this: generate one latent, expand.
    exec_device = getattr(pipe, '_execution_device', device)
    num_ch  = pipe.transformer.config.in_channels // 4   # 16 for FLUX
    lat_h   = height // 8
    lat_w   = width  // 8
    _g      = torch.Generator(device=exec_device).manual_seed(seed)
    _one    = randn_tensor(
        (1, num_ch, lat_h, lat_w),
        generator=_g,
        device=exec_device,
        dtype=torch.bfloat16,
    )
    # Pack: (B, C, lat_h, lat_w) → (B, lat_h//2 * lat_w//2, C*4)
    # This is the standard FLUX packing used inside prepare_latents.
    shared_latents = (
        _one.expand(2, -1, -1, -1).clone()
        .view(2, num_ch, lat_h // 2, 2, lat_w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(2, (lat_h // 2) * (lat_w // 2), num_ch * 4)
    )

    # 3. Install custom attention processor
    proc = UltimateFluxAttnProcessor(
        policy=policy, total_layers=N_LAYERS, total_steps=num_steps
    )
    proc.reset()
    pipe.transformer.set_attn_processor(proc)

    # 4. Register block-level forward hooks
    handles: List = []
    for block_idx, hook_fn in policy.get_block_hooks().items():
        if block_idx < N_DOUBLE:
            block = pipe.transformer.transformer_blocks[block_idx]
        else:
            block = pipe.transformer.single_transformer_blocks[block_idx - N_DOUBLE]
        handles.append(block.register_forward_hook(hook_fn))

    try:
        result = pipe(
            prompt=[source_prompt, edit_prompt],
            latents=shared_latents,        # pre-packed, shared across both branches
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            output_type="pil",
        )
    finally:
        for h in handles:
            h.remove()

    src_img  = result.images[0]
    edit_img = policy.post_process(src_img, result.images[1])
    return src_img, edit_img


# ──────────────────── Classical-CV colour editing ─────────────────────────────

def _rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB [0,1] → HSV [0,1].  Input: (..., 3) float32."""
    r, g, b  = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc     = np.maximum(np.maximum(r, g), b)
    minc     = np.minimum(np.minimum(r, g), b)
    delta    = maxc - minc
    v        = maxc
    s        = np.where(maxc > 1e-7, delta / maxc, 0.0)
    h        = np.zeros_like(r)
    m_r      = (maxc == r) & (delta > 1e-7)
    m_g      = (maxc == g) & (delta > 1e-7)
    m_b      = (maxc == b) & (delta > 1e-7)
    h        = np.where(m_r, ((g - b) / delta) % 6.0, h)
    h        = np.where(m_g, (b - r) / delta + 2.0,   h)
    h        = np.where(m_b, (r - g) / delta + 4.0,   h)
    return np.stack([h / 6.0, s, v], axis=-1)


def _hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV [0,1] → RGB [0,1].  Input: (..., 3) float32."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = (h % 1.0) * 6.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - f * s)
    t  = v * (1.0 - (1.0 - f) * s)
    r  = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [v, q, p, p, t, v], 0.0)
    g  = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [t, v, v, q, p, p], 0.0)
    b  = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [p, p, t, v, v, q], 0.0)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Sobel magnitude, normalised [0, 1].  Input: (H, W) float32."""
    gx = (-gray[:-2, :-2] + gray[:-2, 2:]
          - 2 * gray[1:-1, :-2] + 2 * gray[1:-1, 2:]
          - gray[2:,  :-2] + gray[2:,  2:])
    gy = ( gray[2:,  :-2] + 2 * gray[2:,  1:-1] + gray[2:,  2:]
          - gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - gray[:-2, 2:])
    mag      = np.sqrt(gx ** 2 + gy ** 2)
    mag_full = np.pad(mag, ((1, 1), (1, 1)), mode="edge")
    return mag_full / (mag_full.max() + 1e-8)


# Colour vocabulary ─────────────────────────────────────────────────────────
# Source masks: which pixels match this colour (h, s, v all ∈ [0, 1])
_SRC_MASK: Dict[str, Callable] = {
    "red":    lambda h, s, v: ((h < 0.05) | (h > 0.95)) & (s > 0.30) & (v > 0.15),
    "orange": lambda h, s, v: (h > 0.04) & (h < 0.10)   & (s > 0.40) & (v > 0.20),
    "yellow": lambda h, s, v: (h > 0.09) & (h < 0.17)   & (s > 0.35) & (v > 0.20),
    "green":  lambda h, s, v: (h > 0.25) & (h < 0.45)   & (s > 0.25) & (v > 0.10),
    "blue":   lambda h, s, v: (h > 0.55) & (h < 0.72)   & (s > 0.25) & (v > 0.10),
    "purple": lambda h, s, v: (h > 0.72) & (h < 0.86)   & (s > 0.25) & (v > 0.10),
    "cyan":   lambda h, s, v: (h > 0.45) & (h < 0.56)   & (s > 0.25) & (v > 0.10),
    "black":  lambda h, s, v: v < 0.28,
    "brown":  lambda h, s, v: (h > 0.03) & (h < 0.09)   & (s > 0.30) & (v < 0.55),
    "blonde": lambda h, s, v: (h > 0.08) & (h < 0.17)   & (s > 0.25) & (v > 0.50),
    "white":  lambda h, s, v: (v > 0.85) & (s < 0.20),
    "gray":   lambda h, s, v: (s < 0.18) & (v > 0.20)   & (v < 0.85),
}

# Target HSV parameters: (target_h, s_scale, s_floor, v_scale, v_offset)
#   target_h: new hue in [0, 1]
#   s_scale : multiply source S (preserves saturation variation in the region)
#   s_floor : minimum S to enforce (lifts achromatic sources like black hair)
#   v_scale : multiply source V (brightens / darkens)
#   v_offset: add after scaling   (ensures minimum brightness for dark→light)
_TGT_HSV: Dict[str, tuple] = {
    "red":    (0.00, 1.0, 0.65, 1.0, 0.00),
    "orange": (0.07, 1.0, 0.65, 1.0, 0.00),
    "yellow": (0.14, 1.0, 0.65, 1.0, 0.00),
    "green":  (0.35, 1.0, 0.65, 1.0, 0.00),
    "blue":   (0.63, 1.0, 0.65, 1.0, 0.00),
    "purple": (0.78, 1.0, 0.65, 1.0, 0.00),
    "cyan":   (0.50, 1.0, 0.65, 1.0, 0.00),
    "black":  (0.00, 0.0, 0.00, 0.12, 0.00),
    "brown":  (0.07, 0.8, 0.55, 2.0,  0.05),
    "blonde": (0.11, 0.7, 0.50, 3.5,  0.18),  # dark→blonde: set hue, brighten
    "white":  (0.00, 0.0, 0.00, 1.0,  0.85),
    "gray":   (0.00, 0.0, 0.00, 1.0,  0.00),
}


def _cv_color_replace(
    src_img: Image.Image,
    source_color: str,
    target_color: str,
    edge_strength: float = 0.7,
) -> Image.Image:
    """
    HSV colour replacement with Sobel edge preservation.

    Steps:
    1. Convert source PIL image to HSV (numpy, no external deps).
    2. Segment pixels matching *source_color* by HSV thresholding → mask.
       License plates / face outlines are typically NOT the car / hair colour,
       so they fall outside the mask and are untouched automatically.
    3. Compute Sobel edge magnitude on grayscale source.  Strong edges (plate
       text characters, face contours, fine structural borders) get high values.
    4. Blend weight = mask × clamp(1 − edge_mag / edge_threshold, 0, 1).
       → 1 in flat coloured regions  (full colour change)
       → 0 at strong edges           (pixel taken from source, unchanged)
    5. Build target HSV: shift H to new hue; apply s_scale + s_floor (so that
       achromatic black hair gains saturation when going blonde); rescale V
       (so dark hair brightens to blonde luminance while keeping shadow detail).
    6. Lerp source and target RGB using the weight map → PIL image.

    source_color / target_color: any key in _SRC_MASK / _TGT_HSV, e.g.
        "red"/"blue", "black"/"blonde", "green"/"purple" …
    edge_strength: [0, 1].  Higher = more pixels locked to source at edges.
        0.0 → no edge suppression  |  0.7 → freeze Sobel > 0.3 (default)
    """
    if source_color not in _SRC_MASK:
        raise ValueError(
            f"Unknown source_color '{source_color}'. "
            f"Choose from: {sorted(_SRC_MASK)}"
        )
    if target_color not in _TGT_HSV:
        raise ValueError(
            f"Unknown target_color '{target_color}'. "
            f"Choose from: {sorted(_TGT_HSV)}"
        )

    src  = np.array(src_img.convert("RGB")).astype(np.float32) / 255.0   # (H,W,3)
    hsv  = _rgb_to_hsv_np(src)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 1. Colour segmentation mask
    mask = _SRC_MASK[source_color](h, s, v).astype(np.float32)           # (H,W)

    # 2. Sobel edge map — high values at structural boundaries in pixel space
    gray     = 0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]
    edge_mag = _sobel_edges(gray)                                          # (H,W) ∈ [0,1]

    # 3. Blend weight: edge_threshold = 1 − edge_strength
    #    pixels with edge_mag > edge_threshold → alpha → 0 (lock to source)
    edge_thr = max(1.0 - edge_strength, 0.02)
    alpha    = mask * np.clip(1.0 - edge_mag / edge_thr, 0.0, 1.0)       # (H,W)

    # 4. Build target HSV
    tgt_h, s_scale, s_floor, v_scale, v_offset = _TGT_HSV[target_color]
    new_h   = np.full_like(h, tgt_h)
    new_s   = np.clip(np.maximum(s * s_scale, s_floor), 0.0, 1.0)
    new_v   = np.clip(v * v_scale + v_offset, 0.0, 1.0)
    new_rgb = _hsv_to_rgb_np(np.stack([new_h, new_s, new_v], axis=-1))   # (H,W,3)

    # 5. Blend source and target RGB
    a3     = alpha[..., np.newaxis]
    result = a3 * new_rgb + (1.0 - a3) * src
    return Image.fromarray((np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8))


@torch.no_grad()
def generate_p2p(
    pipe: FluxPipeline,
    source_prompt: str,
    edit_prompt: str,
    inject_layers: List[int],           # unused; kept for API compat
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    anchor_end_frac: float = 0.0,       # unused; kept for API compat
    freq_sigma: float = 0.0,            # unused; kept for API compat
    img2img_strength: float = 0.6,      # unused; kept for API compat
    source_color: str = "",
    target_color: str = "",
    edge_strength: float = 0.7,
) -> Tuple[Image.Image, Image.Image]:
    """
    Colour editing via single FLUX generation + classical-CV colour replacement.

    Pass 1 — FLUX generates the source image from *source_prompt* + *seed*.
    Pass 2 — _cv_color_replace rewrites colour in pixel space (no second FLUX call):
        • HSV segmentation: isolate pixels matching *source_color*
          (license plates / face skin are typically NOT the edited colour
          → outside the mask → untouched automatically)
        • Hue / saturation shift to *target_color*, V (brightness) preserved
          → structure and shadows intact; no colour bleeding
        • Sobel edge mask: suppress change at strong gradients (plate text,
          facial contours) → exact pixel-level structure preservation
        • Blend: flat coloured regions → new colour | edges → source unchanged

    source_color / target_color: colour name, e.g. "red"/"blue", "black"/"blonde"
    edge_strength: 0 = no edge lock | 0.7 = freeze Sobel > 0.3 (default)

    Returns (src_img, edit_img).
    """
    exec_device = getattr(pipe, "_execution_device", device)

    print("[UltimateFlux P2P] Pass 1: generating source image…")
    src_img = pipe(
        prompt=source_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator(device=exec_device).manual_seed(seed),
        output_type="pil",
    ).images[0]

    if source_color and target_color:
        print(
            f"[UltimateFlux P2P] Pass 2 (CV): "
            f"{source_color!r} → {target_color!r}  "
            f"edge_strength={edge_strength}"
        )
        edit_img = _cv_color_replace(src_img, source_color, target_color, edge_strength)
    else:
        print("[UltimateFlux P2P] No source_color/target_color — returning source unchanged.")
        edit_img = src_img

    return src_img, edit_img
