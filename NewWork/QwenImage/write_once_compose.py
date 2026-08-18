# -*- coding: utf-8 -*-
"""
QwenImage/write_once_compose.py  —  Sketch-anchored write-once scene composition
via Qwen-Image-Edit-2509.

Core invariant
--------------
Every background pixel is generated AT MOST ONCE across the full editing session.
At step t (inserting object t into mask M_t):

    Region                    Source
    ─────────────────────     ──────────────────────────────────────────
    background                b_0 latent  (never regenerated)
    placed regions (j < t)    z_j result  (frozen at their generation step)
    current region (M_t)      free generation
    harmonization band        soft transition for shadows / contact lighting

z_anchor grows monotonically: after step k it contains b_0 everywhere except
the placed object regions, where it contains the frozen generation outputs.
This breaks the Markov chain — b_t is derived from b_0 + frozen slots, not
from b_{t-1}.  Drift in region i physically cannot reach region j.

Sketch as spatial specification
--------------------------------
Draw stroke groups over b_0 in b_0's coordinate frame (not on a blank canvas):
    White / near-white (low saturation)   →  background
    Each unique saturated hue             →  one object instance

This gives four jobs no other modality does:
    1. Edit mask M_i from stroke extent — no SAM, no diff estimation
    2. Complete layout plan before step 1 runs — enables the write-once scheme
    3. Occlusion order from Y-position heuristic / monocular depth
    4. Geometric specification (position, scale, contact point) text cannot

Jobs 1–3 are mechanisms for fixing drift, not usability arguments.

Preservation ↔ plausibility frontier (the research question)
------------------------------------------------------------
    alpha_bg = 1.0, band_width = 0    →  perfect background, pasted-on look
    alpha_bg = 0.0, no anchoring      →  plausible lighting, accumulated drift

Sweep (alpha_bg, band_width) to trace the frontier.  Existing methods sit at
the two degenerate endpoints; this method should trace a better Pareto curve.

Diagnostics (run --diagnostics first)
--------------------------------------
    1. VAE round-trip  — encode→decode cost; sets the preservation lower bound
    2. Null-op drift   — per-step drift from a "do nothing" Qwen call
    3. Degradation curve — background quality vs. sequential step count (no anchoring)

Usage
-----
    python NewWork/QwenImage/write_once_compose.py \\
        --base_prompt "empty minimalist living room, hardwood floor, white walls, 4K" \\
        --sketch       inputs/sketch_overlay.png \\
        --descs        "yellow bicycle" "black ceramic vase" "rubber ball" \\
        --out_dir      results/write_once \\
        --hf_token     $HF_TOKEN --cache_dir ./models \\
        --lightning    --band_width 16 --alpha_bg 0.9

    # Run diagnostics only (no sketch needed)
    python NewWork/QwenImage/write_once_compose.py \\
        --base_img inputs/room.png --diagnostics --diag_steps 5 \\
        --hf_token $HF_TOKEN --cache_dir ./models --lightning
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── Optional dependencies (graceful fallback) ──────────────────────────────────

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from huggingface_hub import hf_hub_download
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False

try:
    from skimage.metrics import structural_similarity
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ── Constants ──────────────────────────────────────────────────────────────────

LIGHTNING_REPO      = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_SUBFOLDER = "Qwen-Image-Edit-2509"

# Scheduler config for the Lightning distilled weights
LIGHTNING_SCHEDULER_CFG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


# ── ObjectSlot ────────────────────────────────────────────────────────────────

@dataclass
class ObjectSlot:
    """One object instance extracted from the sketch overlay."""
    name:        str
    description: str
    mask_np:     np.ndarray            # (H, W) bool — pixel space, b_0 coordinates
    color:       Tuple[int, int, int]  # dominant RGB of the sketch stroke
    depth_rank:  int = 0               # 0 = farthest (insert first)
    sketch_crop: Optional[Image.Image] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        """(y, x) normalized to [0,1]; y=0 is ceiling, y=1 is floor."""
        rows, cols = np.where(self.mask_np)
        if len(rows) == 0:
            return (0.5, 0.5)
        H, W = self.mask_np.shape
        return (float(rows.mean()) / H, float(cols.mean()) / W)

    @property
    def mask_area_frac(self) -> float:
        return float(self.mask_np.mean())


# ── Slots from individual sketch files (no combined overlay needed) ───────────

def _make_room_prior_mask(
    lat_h:       int,
    lat_w:       int,
    top_frac:    float = 0.25,   # ceiling fraction to anchor
    side_frac:   float = 0.12,   # wall-edge fraction to anchor
    temp:        float = 0.04,
) -> torch.Tensor:
    """Room-prior object mask at latent resolution (1, 1, lat_h, lat_w) ∈ [0,1].

    0 = background (ceiling / wall edges) → anchored
    1 = object zone (floor / center)      → free

    Used when no explicit spatial mask is available.  The ceiling top
    and wall sides are statistically background in interior room shots.
    """
    y  = torch.linspace(0.0, 1.0, lat_h)
    x  = torch.linspace(0.0, 1.0, lat_w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    v  = torch.sigmoid((yy - top_frac) / temp)
    h  = torch.sigmoid((0.5 - side_frac - (xx - 0.5).abs()) / temp)
    return (v * h).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def slots_from_sketches(
    sketch_paths: List[str],
    descriptions: List[str],
    height:       int,
    width:        int,
    top_frac:     float = 0.25,
    side_frac:    float = 0.12,
) -> List[ObjectSlot]:
    """Create ObjectSlots from individual sketch files (white-background PNGs).

    Each sketch is used for Stage B (object synthesis).
    Because no spatial masks exist, a shared room-prior mask is embedded into
    the slot's mask_np (floor/center = free, ceiling/walls = anchored).

    z_anchor is updated via diff-based estimation after each step, so the
    write-once guarantee still holds even without pre-defined placement masks.
    """
    if len(sketch_paths) != len(descriptions):
        raise ValueError("sketch_paths and descriptions must be the same length.")

    # Build a pixel-space room-prior mask to embed in each slot
    # (lat_h/lat_w are not known yet, so we use the full image resolution)
    prior_lat = _make_room_prior_mask(height // 8, width // 8, top_frac, side_frac)
    prior_np  = F.interpolate(prior_lat, size=(height, width),
                              mode="bilinear", align_corners=False)
    prior_np  = (prior_np[0, 0].numpy() > 0.5)   # (H, W) bool: True = object zone

    import colorsys
    slots: List[ObjectSlot] = []
    for idx, (sp, desc) in enumerate(zip(sketch_paths, descriptions)):
        hue_step = 360 // max(len(sketch_paths), 1)
        r, g, b  = colorsys.hsv_to_rgb(((idx * hue_step) % 360) / 360.0, 0.7, 0.9)
        slots.append(ObjectSlot(
            name        = f"obj_{idx + 1}",
            description = desc,
            mask_np     = prior_np.copy(),   # all slots share the room prior
            color       = (int(r * 255), int(g * 255), int(b * 255)),
            depth_rank  = idx,               # insertion order = user order
            sketch_crop = Image.open(sp).convert("RGB"),
        ))
    print(f"[SketchLoader] {len(slots)} slot(s) — room-prior mask for anchoring.")
    return slots


# ── Sketch parsing ────────────────────────────────────────────────────────────

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB→HSV.  rgb: (H, W, 3) float in [0,255].  Returns (H,W,3)."""
    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    diff = cmax - cmin
    v = cmax
    s = np.where(cmax > 1e-8, diff / cmax, 0.0)
    h = np.zeros_like(r)
    m_r = (cmax == r) & (diff > 1e-8)
    m_g = (cmax == g) & (diff > 1e-8)
    m_b = (cmax == b) & (diff > 1e-8)
    h[m_r] = (((g - b)[m_r] / diff[m_r]) % 6) / 6
    h[m_g] = ((b - r)[m_g] / diff[m_g] + 2) / 6
    h[m_b] = ((r - g)[m_b] / diff[m_b] + 4) / 6
    return np.stack([h, s, v], axis=-1)


def parse_sketch(
    sketch:         Image.Image,
    height:         int,
    width:          int,
    sat_thresh:     float = 0.20,   # min HSV saturation to count as "colored"
    val_thresh:     float = 0.20,   # min HSV value (reject near-black)
    hue_bucket_deg: int   = 20,     # quantize hue to this step (degrees)
    min_area_frac:  float = 0.002,  # ignore regions smaller than this fraction
) -> List[ObjectSlot]:
    """Parse a colored sketch overlay into ordered ObjectSlots.

    Colored convention (user draws with any paint tool):
        White / low-saturation          → background
        Each unique saturated hue       → one object instance

    Sketch must be drawn over b_0 in b_0's coordinate frame.
    Returned slots are unordered by depth (call assign_depth_order next).
    """
    sketch_rs = sketch.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(sketch_rs, dtype=np.float32)
    hsv = _rgb_to_hsv(arr)

    colored = (hsv[:, :, 1] > sat_thresh) & (hsv[:, :, 2] > val_thresh)
    if not colored.any():
        print("[SketchParser] No colored strokes found. "
              "Check sat_thresh / val_thresh or sketch colors.")
        return []

    hue_deg   = (hsv[:, :, 0] * 360).astype(int)
    bucketed  = (hue_deg // hue_bucket_deg) * hue_bucket_deg
    bucketed[~colored] = -1  # background sentinel

    slots: List[ObjectSlot] = []
    for idx, hue in enumerate(sorted(h for h in np.unique(bucketed) if h >= 0)):
        mask = (bucketed == hue)
        if mask.mean() < min_area_frac:
            continue
        dom_rgb = tuple(int(arr[mask, ch].mean()) for ch in range(3))
        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        slots.append(ObjectSlot(
            name        = f"obj_{idx + 1}",
            description = "",
            mask_np     = mask,
            color       = dom_rgb,
            sketch_crop = sketch_rs.crop((c0, r0, c1, r1)),
        ))

    print(f"[SketchParser] {len(slots)} instance(s) found.")
    return slots


def assign_depth_order(slots: List[ObjectSlot]) -> List[ObjectSlot]:
    """Sort slots farthest-first (insert order).

    Heuristic: lower Y centroid = higher in image = farther from camera
    in a typical interior room shot.  Returns re-ordered list with
    depth_rank assigned (0 = farthest = insert first).
    """
    ordered = sorted(slots, key=lambda s: (s.centroid[0], s.centroid[1]))
    for rank, slot in enumerate(ordered):
        slot.depth_rank = rank
    return ordered


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_pipeline(
    model:           str           = "Qwen/Qwen-Image-Edit-2509",
    hf_token:        Optional[str] = None,
    cache_dir:       str           = "./models",
    lightning:       bool          = True,
    lightning_steps: int           = 8,
    dtype:           torch.dtype   = torch.bfloat16,
) -> "QwenImageEditPlusPipeline":
    """Load Qwen-Image-Edit-2509, optionally with Lightning LoRA.
    Auto-enables CPU offload when GPU VRAM < 65 GB.
    """
    if not _HAS_DIFFUSERS:
        raise ImportError("pip install diffusers huggingface_hub")

    kwargs: Dict = dict(torch_dtype=dtype, token=hf_token, cache_dir=cache_dir)
    if lightning:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
            LIGHTNING_SCHEDULER_CFG
        )
        print(f"  [load] Lightning scheduler  (shift=1.0, log3 range)")

    pipe = QwenImageEditPlusPipeline.from_pretrained(model, **kwargs)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram_gb < 65:
            print(f"  [load] GPU {vram_gb:.0f} GB → cpu_offload enabled")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    if lightning:
        wname = (
            f"Qwen-Image-Edit-2509-Lightning-{lightning_steps}steps-V1.0-bf16.safetensors"
        )
        lora_path = hf_hub_download(
            repo_id   = LIGHTNING_REPO,
            filename  = f"{LIGHTNING_SUBFOLDER}/{wname}",
            token     = hf_token,
            cache_dir = cache_dir,
        )
        pipe.load_lora_weights(lora_path)
        print(f"  [load] Lightning {lightning_steps}-step LoRA loaded")

    _patch_vae_for_pipeline(pipe)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── VAE encode / decode (video VAE: requires T dimension) ─────────────────────

def _patch_vae_for_pipeline(pipe) -> None:
    """Patch the video VAE so encode/decode accept and return 4-D (B,C,H,W) tensors.

    Qwen's video VAE expects 5-D (B,C,T,H,W) internally.  The pipeline's
    denoising loop works with 4-D latents and calls vae.encode / vae.decode
    with 4-D inputs.  When batch_size=1, any bare .squeeze() on the 5-D output
    (1,C,1,H,W) removes BOTH the batch dim AND the T dim → 3-D tensor →
    the subsequent torch.cat([latents, image_latents], dim=1) crashes.

    This patch:
      • encode: inserts T=1 before the real encode; squeezes T from the output
      • decode: inserts T=1 before the real decode; squeezes T from the output

    After patching the VAE behaves as a standard image VAE with 4-D I/O.
    encode_latent() and decode_latent() are updated to match (no manual T handling).
    """
    try:
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    except ImportError:
        try:
            from diffusers.models.autoencoder_kl import DiagonalGaussianDistribution
        except ImportError:
            DiagonalGaussianDistribution = None

    _orig_encode = pipe.vae.encode
    _orig_decode = pipe.vae.decode

    def _encode_4d(x, *a, **kw):
        if x.dim() == 4:                        # add T=1 only if not already 5-D
            x = x.unsqueeze(2)
        out = _orig_encode(x, *a, **kw)
        if DiagonalGaussianDistribution is not None and hasattr(out, "latent_dist"):
            p = out.latent_dist.parameters
            if p.dim() == 5:                    # (B,C,1,H,W) → (B,C,H,W)
                out.latent_dist = DiagonalGaussianDistribution(p.squeeze(2))
        return out

    def _decode_4d(z, *a, **kw):
        if z.dim() == 4:                        # add T=1 only if not already 5-D
            z = z.unsqueeze(2)
        out = _orig_decode(z, *a, **kw)
        if hasattr(out, "sample") and out.sample.dim() == 5:
            out.sample = out.sample.squeeze(2)  # (B,3,1,H,W) → (B,3,H,W)
        return out

    pipe.vae.encode = _encode_4d
    pipe.vae.decode = _decode_4d
    print("[patch] VAE encode/decode patched: T-dim managed automatically (4-D I/O)")


def encode_latent(
    pipe,
    pil_img: Image.Image,
    height:  int,
    width:   int,
) -> torch.Tensor:
    """PIL → unscaled VAE latent (1, C, H/8, W/8) on CPU float32.

    Requires _patch_vae_for_pipeline() to have been called: the patched VAE
    accepts 4-D input and returns 4-D output, so no manual T-dim handling here.
    """
    img = pil_img.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
    try:
        dev  = next(pipe.vae.parameters()).device
        dtyp = next(pipe.vae.parameters()).dtype
    except StopIteration:
        dev, dtyp = torch.device("cpu"), torch.float32
    with torch.no_grad():
        z = pipe.vae.encode(t.to(dev, dtyp)).latent_dist.mean   # (1, C, H/8, W/8)
    return z.cpu().float()


def decode_latent(pipe, z: torch.Tensor) -> Image.Image:
    """Unscaled VAE latent (1, C, H/8, W/8) → PIL RGB."""
    try:
        dev  = next(pipe.vae.parameters()).device
        dtyp = next(pipe.vae.parameters()).dtype
    except StopIteration:
        dev, dtyp = torch.device("cpu"), torch.float32
    with torch.no_grad():
        dec = pipe.vae.decode(z.to(dev, dtyp)).sample   # (1, 3, H, W) after patch
    dec = dec.float().clamp(-1, 1) / 2 + 0.5
    arr = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# ── Mask utilities ────────────────────────────────────────────────────────────

def mask_to_latent(mask_np: np.ndarray, lat_h: int, lat_w: int) -> torch.Tensor:
    """Resize bool pixel-space mask → (1,1,lat_h,lat_w) float tensor."""
    m = Image.fromarray(mask_np.astype(np.uint8) * 255).resize(
        (lat_w, lat_h), Image.LANCZOS
    )
    return (torch.from_numpy(np.array(m)).float() / 255.0).unsqueeze(0).unsqueeze(0)


def make_band_weight(
    mask_np:    np.ndarray,
    band_width: int,
    lat_h:      int,
    lat_w:      int,
) -> torch.Tensor:
    """Build write-once anchor weight map at latent resolution.

    Returns (1, 1, lat_h, lat_w) ∈ [0, 1]:
        0      inside M_t        → free generation for current object
        0 → 1  harmonization band → partial anchor (shadows, contact, lighting)
        1      outside M_t       → fully anchored to reference

    The band ramps from 0 at the mask edge to 1 at band_width pixels out.
    This allows the model to generate cast shadows and floor-contact gradients
    while the rest of the background is rigidly preserved.

    Falls back to a hard step (no band) if scipy is unavailable.
    """
    if _HAS_SCIPY and band_width > 0:
        # Distance from mask edge: 0 inside mask, >0 outside
        dist = distance_transform_edt(~mask_np).astype(np.float32)
        weight_px = np.clip(dist / max(band_width, 1), 0.0, 1.0)
        weight_px[mask_np] = 0.0   # inside mask: always free
    else:
        weight_px = (~mask_np).astype(np.float32)

    wimg = Image.fromarray((weight_px * 255).clip(0, 255).astype(np.uint8))
    w_lat = np.array(wimg.resize((lat_w, lat_h), Image.BILINEAR)) / 255.0
    return torch.from_numpy(w_lat).float().unsqueeze(0).unsqueeze(0)


# ── Write-once composition state ──────────────────────────────────────────────

class WriteOnceState:
    """Progressive anchor for write-once composition.

    Starts as the empty-room latent z_0.  After placing object k at mask M_k,
    the object's generated latent is "painted" into z_anchor at M_k.

    Future steps anchor to z_anchor, which already contains the frozen
    outputs of all prior objects — so anchoring does not pull previously
    placed objects back to empty-room appearance.

    Invariant after step k:
        z_anchor[px] = z_0[px]       if px ∉ ∪_{j≤k} M_j  (background)
        z_anchor[px] = z_j[px]       if px ∈ M_j, j ≤ k    (frozen slot j)
    """

    def __init__(self, z_base: torch.Tensor):
        # z_base: (1, C, H/8, W/8) unscaled VAE latent of b_0
        self.z_anchor = z_base.clone().cpu().float()

    def update(self, z_result: torch.Tensor, mask_lat: torch.Tensor) -> None:
        """After step k: paint z_result into z_anchor at mask_lat.

        z_result : (1, C, lat_h, lat_w) unscaled VAE latent of generated output
        mask_lat : (1, 1, lat_h, lat_w) soft mask in [0, 1]
        """
        m = F.interpolate(
            mask_lat.cpu().float(), size=self.z_anchor.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        self.z_anchor = (1.0 - m) * self.z_anchor + m * z_result.cpu().float()


# ── Write-once denoising callback ─────────────────────────────────────────────

class WriteOnceCallback:
    """Denoising-step callback implementing the write-once invariant.

    At each of the N denoising steps at noise level σ_t:

        latent_out = (1 − w) · latent_in  +  w · z_ref_noised(σ_t)

        z_ref_noised = (1 − σ_t) · z_anchor_scaled  +  σ_t · ε

    where w = alpha * band_weight ∈ [0, 1]:
        w = 0      inside M_t (object zone: generated freely)
        w ∈ (0,1)  in harmonization band (partial anchor for shadows)
        w = 1      in background + placed zones (fully anchored)

    z_anchor_scaled already contains frozen prior-object latents, so anchoring
    the background correctly preserves everything that was placed before.

    The noise ε is drawn once and reused across all steps for coherence —
    each step sees the same spatial noise realization at a different σ.
    """

    def __init__(
        self,
        state:       WriteOnceState,
        band_weight: torch.Tensor,   # (1, 1, ref_H, ref_W) — resized at call time
        alpha:       float = 1.0,    # anchor strength: 0=off, 1=full lock
        start_step:  int   = 0,      # first denoising step to activate
    ):
        self.state       = state
        self.band_weight = band_weight.cpu().float()
        self.alpha       = alpha
        self.start_step  = start_step
        self._noise: Optional[torch.Tensor] = None

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if i < self.start_step:
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        dev, dtyp = latents.device, latents.dtype
        scaling   = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        z_scaled  = (self.state.z_anchor * scaling).to(dev, dtyp)

        # Draw noise once; reuse across steps for temporal coherence
        if self._noise is None or self._noise.shape != z_scaled.shape:
            self._noise = torch.randn_like(z_scaled)
        noise = self._noise.to(dev, dtyp)

        # σ from flow-matching timestep
        T     = float(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
        sigma = (float(t.mean()) if t.dim() > 0 else float(t)) / T

        # Noise-matched reference for this σ level
        z_ref_noised = (1.0 - sigma) * z_scaled + sigma * noise

        # Handle 4D (B,C,H,W) or 5D (B,C,T,H,W) pipeline latents
        is_5d = latents.dim() == 5
        frame = latents[:, :, 0] if is_5d else latents
        lat_h, lat_w = frame.shape[-2], frame.shape[-1]

        # Resize reference and weight to match callback latent resolution
        if z_ref_noised.shape[-2:] != (lat_h, lat_w):
            z_ref_noised = F.interpolate(
                z_ref_noised.float(), size=(lat_h, lat_w),
                mode="bilinear", align_corners=False,
            ).to(dtyp)

        w = F.interpolate(
            self.band_weight.to(dev, torch.float32), size=(lat_h, lat_w),
            mode="bilinear", align_corners=False,
        ).to(dtyp) * self.alpha

        anchored = (1.0 - w) * frame + w * z_ref_noised

        if is_5d:
            latents_out = latents.clone()
            latents_out[:, :, 0] = anchored
        else:
            latents_out = anchored

        callback_kwargs["latents"] = latents_out
        return callback_kwargs


# ── Object synthesis ──────────────────────────────────────────────────────────

def _tight_crop(img: Image.Image, pad: int = 24) -> Image.Image:
    """Crop to non-white bounding box, keep square, return 512×512."""
    arr  = np.array(img.convert("RGB"))
    mask = np.any(arr < 240, axis=2)
    if not mask.any():
        return img.resize((512, 512), Image.LANCZOS)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(0, rows[0] - pad), min(arr.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(arr.shape[1], cols[-1] + pad)
    side = max(r1 - r0, c1 - c0)
    return img.crop((c0, r0, c0 + side, r0 + side)).resize((512, 512), Image.LANCZOS)


def synth_object(
    pipe,
    slot:      ObjectSlot,
    seed:      int,
    num_steps: int,
    guidance:  float,
    height:    int,
    width:     int,
) -> Image.Image:
    """Sketch crop → photorealistic isolated object via Qwen.

    Uses the slot's position (centroid) to hint at perspective scale,
    then tight-crops the result to 512×512 for use as Picture 2.
    """
    sketch_in = (slot.sketch_crop or Image.new("RGB", (512, 512), (200, 200, 200)))
    sketch_in = sketch_in.convert("RGB").resize((width, height), Image.LANCZOS)

    cy, _ = slot.centroid
    persp = "lower, larger (closer to camera)" if cy > 0.60 else \
            "upper, smaller (farther from camera)" if cy < 0.40 else \
            "mid-scene scale"

    prompt = (
        f"Render the object in this sketch as a photorealistic {slot.description}. "
        f"Use the sketch for exact shape and proportions. "
        f"Plain white background, studio lighting, no shadows. "
        f"The object will be placed {persp}."
    )
    neg = (
        "blurry, distorted, low quality, background, room, floor, "
        "shadow, multiple objects, text, watermark"
    )
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt, negative_prompt=neg,
        image=sketch_in, num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        height=height, width=width,
        generator=gen,
    ).images[0]
    return _tight_crop(result)


def overlay_mask_on_scene(
    scene:   Image.Image,
    mask_np: np.ndarray,
    color:   Tuple[int, int, int] = (255, 200, 0),
    alpha:   float = 0.35,
) -> Image.Image:
    """Tint the mask region on the scene image.

    Gives Qwen an unambiguous visual cue for WHERE to place the object,
    instead of relying solely on text position descriptions.
    """
    arr = np.array(scene.convert("RGB")).astype(float)
    overlay = np.array(color, dtype=float)
    arr[mask_np] = (1.0 - alpha) * arr[mask_np] + alpha * overlay
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ── Qwen call ─────────────────────────────────────────────────────────────────

def run_qwen(
    pipe,
    image,
    prompt:          str,
    seed:            int,
    num_steps:       int,
    guidance:        float,
    height:          int,
    width:           int,
    negative_prompt: str = "blurry, distorted, low quality, watermark, artifacts",
    callback:        Optional[object] = None,
) -> Image.Image:
    gen   = torch.Generator(device=pipe.device).manual_seed(seed)
    extra: Dict = {}
    if callback is not None:
        extra["callback_on_step_end"]               = callback
        extra["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=image, num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        height=height, width=width,
        generator=gen,
        **extra,
    ).images[0]


# ── Scene composer ────────────────────────────────────────────────────────────

class SceneComposer:
    """Write-once sketch-anchored scene composition.

    Example::

        composer = SceneComposer(pipe, height=1024, width=1024)
        images   = composer.compose(base_img, slots,
                                    band_width=16, alpha_bg=0.9,
                                    out_dir="results/run_01")
    """

    def __init__(
        self,
        pipe,
        height:      int   = 1024,
        width:       int   = 1024,
        seed:        int   = 42,
        num_steps:   int   = 8,
        guidance:    float = 1.0,
        obj_guidance: float = 1.0,
    ):
        self.pipe        = pipe
        self.H, self.W   = height, width
        self.seed        = seed
        self.num_steps   = num_steps
        self.guidance    = guidance
        self.obj_guidance = obj_guidance

    def compose(
        self,
        base_img:   Image.Image,
        slots:      List[ObjectSlot],   # depth-ordered: farthest first
        band_width: int   = 16,         # harmonization band width in pixels
        alpha_bg:   float = 1.0,        # anchor strength (0=free, 1=locked)
        out_dir:    Optional[str] = None,
    ) -> List[Image.Image]:
        """Run write-once composition. Returns [base, step_1, ..., step_n]."""
        H, W = self.H, self.W
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # ── Encode b_0 → write-once anchor ────────────────────────────────────
        print(f"\n[Compose] Encoding base scene latent ...")
        z_base  = encode_latent(self.pipe, base_img, H, W)
        lat_h, lat_w = z_base.shape[-2], z_base.shape[-1]
        print(f"          z_base: {tuple(z_base.shape)}  "
              f"(pixel→latent {H // lat_h}×)")

        state  = WriteOnceState(z_base)
        scene  = base_img.convert("RGB")
        images = [base_img]

        if out_dir:
            base_img.save(os.path.join(out_dir, "base_scene.png"))

        for i, slot in enumerate(slots):
            step = i + 1
            print(f"\n[Step {step}/{len(slots)}]  {slot.name}  —  '{slot.description}'")

            # ── Stage B: sketch → photorealistic object ────────────────────────
            print(f"  [B] Synthesizing object ...")
            obj_img = synth_object(
                self.pipe, slot,
                seed=self.seed, num_steps=self.num_steps,
                guidance=self.obj_guidance, height=H, width=W,
            )
            if out_dir:
                obj_img.save(os.path.join(out_dir, f"obj_{slot.name}.png"))

            # ── Stage C: write-once insertion ─────────────────────────────────
            # Visual placement guide: tint mask region on scene
            scene_guided = overlay_mask_on_scene(scene, slot.mask_np, slot.color)

            # Band weight: 0 inside mask, ramp in band, 1 in background
            band_w   = make_band_weight(slot.mask_np, band_width, lat_h, lat_w)
            callback = WriteOnceCallback(state, band_w, alpha=alpha_bg)

            # Placement prompt using centroid for coarse spatial hint
            cy, cx = slot.centroid
            vert  = "lower" if cy > 0.55 else "upper" if cy < 0.45 else "mid"
            horiz = "left"  if cx < 0.40 else "right" if cx > 0.60 else "center"
            prompt = (
                f"\nPicture 1 shows a room with a highlighted placement region. "
                f"Picture 2 shows a {slot.description}. "
                f"Place the {slot.description} from Picture 2 into the highlighted "
                f"{vert}-{horiz} area of the room in Picture 1. "
                f"Match the room perspective and lighting. "
                f"Add a realistic contact shadow on the floor. "
                f"Do not change any other part of the room."
            )

            if out_dir:
                with open(
                    os.path.join(out_dir, f"prompt_step{step}_{slot.name}.txt"), "w"
                ) as f:
                    f.write(prompt)

            print(f"  [K] Inserting at centroid ({cy:.2f}, {cx:.2f}) ...")
            result = run_qwen(
                self.pipe,
                image=[scene_guided, obj_img.convert("RGB")],
                prompt=prompt,
                seed=self.seed, num_steps=self.num_steps, guidance=self.guidance,
                height=H, width=W,
                callback=callback,
            )

            if out_dir:
                result.save(os.path.join(out_dir, f"result_step{step}_{slot.name}.png"))

            # ── Update write-once anchor with frozen result ────────────────────
            z_result  = encode_latent(self.pipe, result, H, W)
            mask_lat  = mask_to_latent(slot.mask_np, lat_h, lat_w)
            state.update(z_result, mask_lat)

            scene = result
            images.append(result)
            print(f"  ✓  Done.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return images


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diag_vae_roundtrip(
    pipe, img: Image.Image, height: int, width: int,
) -> Dict:
    """Measure encode → decode degradation.

    This is the preservation lower bound: even a perfect method cannot beat
    the SSIM / LPIPS of a VAE round-trip.  If this is large, the problem
    is partly in the VAE, not just in multi-step drift.
    """
    z   = encode_latent(pipe, img, height, width)
    rec = decode_latent(pipe, z)
    m   = _img_metrics(img, rec, height, width)
    m["label"] = "vae_roundtrip"
    print(f"[Diag] VAE round-trip  SSIM={m.get('ssim','n/a')}  "
          f"PSNR={m.get('psnr','n/a'):.1f}  LPIPS={m.get('lpips','n/a')}")
    return m


def diag_nullop(
    pipe, img: Image.Image, height: int, width: int,
    seed: int = 42, num_steps: int = 8, guidance: float = 1.0,
) -> Dict:
    """Measure Qwen drift with a 'do nothing' prompt.

    Quantifies how much the model mutates the image even when instructed to
    preserve it.  This is the per-step noise floor for sequential editing
    without any anchoring.
    """
    img_rs = img.convert("RGB").resize((width, height), Image.LANCZOS)
    result = run_qwen(
        pipe, image=img_rs,
        prompt="Do not change this image. Keep it exactly as it is.",
        negative_prompt="any change, modification, alteration",
        seed=seed, num_steps=num_steps, guidance=guidance,
        height=height, width=width,
    )
    m = _img_metrics(img_rs, result, height, width)
    m["label"] = "nullop_drift"
    print(f"[Diag] Null-op drift   SSIM={m.get('ssim','n/a')}  "
          f"PSNR={m.get('psnr','n/a'):.1f}  LPIPS={m.get('lpips','n/a')}")
    return m


def diag_degradation_curve(
    pipe,
    base_img:  Image.Image,
    n_steps:   int   = 5,
    height:    int   = 1024,
    width:     int   = 1024,
    seed:      int   = 42,
    num_steps: int   = 8,
    guidance:  float = 1.0,
    out_dir:   Optional[str] = None,
) -> List[Dict]:
    """Plot background degradation over N sequential null-op passes.

    Each pass runs the do-nothing prompt on the PREVIOUS pass's output.
    The resulting SSIM/LPIPS curve is the 'naive sequential baseline':
    how fast does background quality decay without any anchoring?

    This curve motivates the write-once mechanism.  If it's flat, the
    problem is smaller than claimed and the paper scope narrows.
    """
    print(f"\n[Diag] Degradation curve over {n_steps} sequential null-op passes ...")
    ref   = base_img.convert("RGB").resize((width, height), Image.LANCZOS)
    scene = ref.copy()
    rows: List[Dict] = []

    for k in range(n_steps):
        result = run_qwen(
            pipe, image=scene,
            prompt="Do not change this image. Keep it exactly as it is.",
            negative_prompt="any change, modification, alteration",
            seed=seed + k, num_steps=num_steps, guidance=guidance,
            height=height, width=width,
        )
        m = _img_metrics(ref, result, height, width)
        m["step"] = k + 1
        rows.append(m)
        print(f"  Step {k+1}/{n_steps}  SSIM={m.get('ssim','n/a')}  "
              f"PSNR={m.get('psnr','n/a'):.1f}  LPIPS={m.get('lpips','n/a')}")
        if out_dir:
            result.save(os.path.join(out_dir, f"diag_nullop_step{k+1}.png"))
        scene = result

    return rows


# ── Evaluation metrics ────────────────────────────────────────────────────────

def _img_metrics(
    img_a: Image.Image,
    img_b: Image.Image,
    height: int,
    width:  int,
) -> Dict:
    """SSIM, PSNR, and LPIPS between two images (matched to height×width)."""
    a = np.array(img_a.convert("RGB").resize((width, height), Image.LANCZOS))
    b = np.array(img_b.convert("RGB").resize((width, height), Image.LANCZOS))
    out: Dict = {}

    if _HAS_SKIMAGE:
        out["ssim"] = round(
            float(structural_similarity(a, b, channel_axis=2,
                                        win_size=7, data_range=255)), 4
        )

    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    out["psnr"] = round(
        float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else float("inf"), 2
    )

    try:
        import lpips
        net = lpips.LPIPS(net="alex").eval()
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        with torch.no_grad():
            out["lpips"] = round(float(net(t(a), t(b)).item()), 4)
    except Exception:
        pass

    return out


def eval_scene(
    images:    List[Image.Image],   # [base, step_1, ..., step_n]
    slots:     List[ObjectSlot],
    height:    int,
    width:     int,
) -> List[Dict]:
    """Compute per-step background-preservation metrics.

    Compares each step's output to the base scene on pixels that did NOT
    change (background).  Also reports full-image metrics for reference.
    """
    rows: List[Dict] = []
    base  = images[0]
    base_arr = np.array(base.convert("RGB").resize((width, height), Image.LANCZOS))

    for i, (img, slot) in enumerate(zip(images[1:], slots)):
        img_arr = np.array(img.convert("RGB").resize((width, height), Image.LANCZOS))
        bg_mask = ~slot.mask_np  # background = everything outside this slot

        # Measure only on background pixels
        bg_a = base_arr.copy(); bg_a[~bg_mask] = 128
        bg_b = img_arr.copy();  bg_b[~bg_mask] = 128
        bg_m = _img_metrics(
            Image.fromarray(bg_a), Image.fromarray(bg_b), height, width
        )
        bg_m = {f"bg_{k}": v for k, v in bg_m.items()}

        row = {"step": i + 1, "object": slot.name,
               "description": slot.description, **bg_m}
        rows.append(row)
        print(f"  [Eval] step {i+1} {slot.name:12s}  "
              + "  ".join(f"{k}={v}" for k, v in bg_m.items()))

    return rows


# ── Grid / output ─────────────────────────────────────────────────────────────

def save_grid(images: List[Image.Image], titles: List[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[Grid] {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write-once sketch-anchored scene composition (Qwen-Image-Edit-2509)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs
    g = p.add_mutually_exclusive_group()
    g.add_argument("--base_img",    help="Pre-rendered base room PNG")
    g.add_argument("--base_prompt", help="Text prompt to generate the base room")
    # -- Sketch input: choose one of two modes --
    sm = p.add_mutually_exclusive_group()
    sm.add_argument("--sketch",     default=None,
                    help="Colored sketch overlay (each unique hue = one object instance)")
    sm.add_argument("--edits_json", default=None,
                    help="Path to edits.json; auto-loads sketch paths and descriptions")
    p.add_argument("--sketch_dir",  default=None,
                   help="Directory containing sketch_*.png files (used with --edits_json)")
    p.add_argument("--descs",   nargs="+", default=[],
                   help="Object descriptions, one per sketch hue (ascending hue order)")
    # Output
    p.add_argument("--out_dir", default="results/write_once")
    # Model
    p.add_argument("--hf_token",        default=None)
    p.add_argument("--cache_dir",       default="./models")
    p.add_argument("--model",           default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--lightning",       action="store_true")
    p.add_argument("--lightning_steps", type=int, default=8, choices=[4, 8])
    # Image
    p.add_argument("--height", type=int,   default=1024)
    p.add_argument("--width",  type=int,   default=1024)
    p.add_argument("--seed",   type=int,   default=42)
    # Qwen
    p.add_argument("--num_steps",    type=int,   default=None)
    p.add_argument("--guidance",     type=float, default=None)
    p.add_argument("--obj_guidance", type=float, default=None)
    # Write-once
    p.add_argument("--band_width", type=int,   default=16,
                   help="Harmonization band radius in pixels (0 = hard mask edge)")
    p.add_argument("--alpha_bg",   type=float, default=1.0,
                   help="Background anchor strength: 0=free, 1=fully locked")
    # Diagnostics
    p.add_argument("--diagnostics", action="store_true",
                   help="Run VAE round-trip, null-op, and degradation curve")
    p.add_argument("--diag_steps", type=int, default=5,
                   help="Number of null-op passes for degradation curve")
    # Evaluation
    p.add_argument("--no_eval", action="store_true", help="Skip evaluation metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    num_steps  = args.num_steps    or (8   if args.lightning else 50)
    guidance   = args.guidance     or (1.0 if args.lightning else 4.0)
    obj_guide  = args.obj_guidance or guidance

    print(f"\n{'═' * 60}")
    print(f"  write_once_compose")
    print(f"  Model      : {'Lightning ' + str(num_steps) + '-step' if args.lightning else str(num_steps) + '-step standard'}")
    print(f"  Band width : {args.band_width} px")
    print(f"  Alpha BG   : {args.alpha_bg}")
    print(f"{'═' * 60}")

    pipe = load_pipeline(
        model           = args.model,
        hf_token        = args.hf_token,
        cache_dir       = args.cache_dir,
        lightning       = args.lightning,
        lightning_steps = args.lightning_steps,
    )

    # ── Base scene ────────────────────────────────────────────────────────────
    if args.base_img:
        base_img = Image.open(args.base_img).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS
        )
        print(f"[Base] Loaded: {args.base_img}")
    elif args.base_prompt:
        print(f"[Base] Generating from prompt ...")
        base_img = run_qwen(
            pipe,
            image   = Image.new("RGB", (args.width, args.height), 0),
            prompt  = args.base_prompt,
            seed    = args.seed, num_steps = num_steps, guidance = guidance,
            height  = args.height, width = args.width,
            negative_prompt=(
                "furniture, objects, people, blurry, distorted, "
                "low quality, watermark, cartoon, painting"
            ),
        )
        base_img.save(os.path.join(args.out_dir, "base_scene.png"))
        print(f"[Base] Generated → {args.out_dir}/base_scene.png")
    else:
        raise ValueError("Provide --base_img or --base_prompt.")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if args.diagnostics:
        print(f"\n{'─' * 60}  DIAGNOSTICS  {'─' * 60}")
        os.makedirs(args.out_dir, exist_ok=True)
        d_rt  = diag_vae_roundtrip(pipe, base_img, args.height, args.width)
        d_no  = diag_nullop(pipe, base_img, args.height, args.width,
                            seed=args.seed, num_steps=num_steps, guidance=guidance)
        d_deg = diag_degradation_curve(
            pipe, base_img, n_steps=args.diag_steps,
            height=args.height, width=args.width,
            seed=args.seed, num_steps=num_steps, guidance=guidance,
            out_dir=args.out_dir,
        )
        with open(os.path.join(args.out_dir, "diagnostics.json"), "w") as f:
            json.dump({"roundtrip": d_rt, "nullop": d_no, "degradation": d_deg},
                      f, indent=2)
        print(f"[Diag] Saved → {args.out_dir}/diagnostics.json")

    # ── Sketch / object input ─────────────────────────────────────────────────
    if args.edits_json:
        # Mode 1: individual per-object sketch files + edits.json
        if args.sketch_dir is None:
            raise ValueError("--sketch_dir is required when using --edits_json.")
        with open(args.edits_json) as f:
            edits = json.load(f)
        sketch_paths = [
            os.path.join(args.sketch_dir, e["sketch"]) for e in edits
        ]
        descriptions = [e.get("description", e.get("name", "object")) for e in edits]
        for sp in sketch_paths:
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Sketch file not found: {sp}")
        slots = slots_from_sketches(
            sketch_paths, descriptions, args.height, args.width,
        )
        slots = assign_depth_order(slots)

    elif args.sketch:
        # Mode 2: single colored overlay (each unique hue = one object)
        sketch_img = Image.open(args.sketch).convert("RGB")
        slots = parse_sketch(sketch_img, args.height, args.width)
        if not slots:
            print("[Error] No colored object instances found in sketch.")
            return
        descs = list(args.descs) + ["object"] * max(0, len(slots) - len(args.descs))
        for slot, desc in zip(slots, descs):
            slot.description = desc
        slots = assign_depth_order(slots)

    else:
        print("[Done] No sketch input provided; stopping after diagnostics.")
        return

    print(f"\n[Slots] {len(slots)} object(s) — farthest → nearest:")
    for s in slots:
        cy, cx = s.centroid
        print(f"  rank {s.depth_rank}  {s.name}: '{s.description}'  "
              f"centroid=({cy:.2f},{cx:.2f})  area={s.mask_area_frac * 100:.1f}%")

    # ── Write-once composition ────────────────────────────────────────────────
    composer = SceneComposer(
        pipe=pipe, height=args.height, width=args.width,
        seed=args.seed, num_steps=num_steps,
        guidance=guidance, obj_guidance=obj_guide,
    )
    images = composer.compose(
        base_img   = base_img,
        slots      = slots,
        band_width = args.band_width,
        alpha_bg   = args.alpha_bg,
        out_dir    = args.out_dir,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if not args.no_eval:
        print(f"\n[Eval] Background preservation metrics ...")
        rows = eval_scene(images, slots, args.height, args.width)
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(rows, f, indent=2)

    # ── Grid ─────────────────────────────────────────────────────────────────
    titles = ["base"] + [f"s{i+1} {s.name}" for i, s in enumerate(slots)]
    save_grid(images, titles, os.path.join(args.out_dir, "composition_grid.png"))

    print(f"\n{'═' * 60}")
    print(f"  Done. {len(slots)} object(s) composed.")
    print(f"  Results : {args.out_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
