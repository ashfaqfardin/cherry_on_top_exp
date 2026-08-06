# -*- coding: utf-8 -*-
"""
phase1_collage_qwen.py — AnyDoor's Collage Method with Qwen-Image-Edit-2509

Port of phase1_collage_kontext.py replacing FLUX.1-Kontext-dev with
Qwen/Qwen-Image-Edit-2509  (QwenImageEditPlusPipeline, diffusers).

Key differences from the FLUX version
---------------------------------------
  Model       : Qwen/Qwen-Image-Edit-2509 (20.4B MMDiT, 60 dual-stream blocks)
  VAE         : Wan2.1-based, 8× + 2×2 packing → effective stride 16 px/token
  Conditioning: Qwen is image-to-image — the collage IS the starting image,
                not a separate reference stream (no [gen|ref] token split).
  Stage A     : No FLUX LoRA — Qwen renders the sketch directly via text prompt.
  K/V inject  : Deferred. Qwen has no separate ref-token slice to pull K/V from.
                The collage naturally conditions the edit.
  BCG callback: VAE shift/scale read dynamically from pipe.vae.config.
  All CV/PIL  : COL, BBOX, MASK, pixel BCG — identical to the Kontext version.

Pipeline
---------
  Stage A  : sketch + description → Qwen(image=sketch, prompt) → obj_img
  Stage BBOX: placement mask → bbox  (mask_{name}.png required)
  Stage MASK: threshold obj_img → ref_mask
  Stage COL : paste obj_img into scene → collage_scene
  Stage K  : Qwen(image=collage_scene, prompt=blend_prompt) + BLD latent BCG
  Stage BCG : pixel-space tight-mask composite to restore background

Usage
------
  python NewWork/KontextEval/phase1_collage_qwen.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token   $HF_TOKEN \\
      --cache_dir  ./models \\
      --out_dir    results/phase1_qwen

  Add --cpu_offload if VRAM < 45 GB (FP16) or use a quantised checkpoint.

Key flags
----------
  --guidance        float  CFG scale for Stage K.  Default 3.5
  --lora_guidance   float  CFG scale for Stage A.  Default 3.5
  --collage_mode    full | sobel | blend
  --num_steps       int    Inference steps.         Default 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImageEditPlusPipeline
from PIL import Image


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_qwen_pipeline(
    model_path: str = "Qwen/Qwen-Image-Edit-2509",
    hf_token: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "./models",
    cpu_offload: bool = False,
) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path, torch_dtype=dtype, token=hf_token, cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Qwen inference ────────────────────────────────────────────────────────────

def run_qwen(
    pipe, canvas: Image.Image, prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    bcg_callback=None,
) -> Image.Image:
    # Qwen uses true_cfg_scale (classifier-free guidance), not guidance_scale.
    # A negative_prompt must be provided to activate CFG; without it the
    # guidance_scale param is silently ignored.
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    cb_kwargs: dict = {}
    if bcg_callback is not None:
        cb_kwargs["callback_on_step_end"] = bcg_callback
        cb_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
        image=canvas,
        num_inference_steps=num_steps, true_cfg_scale=guidance,
        height=height, width=width, generator=generator,
        **cb_kwargs,
    ).images[0]


# ── Result grid ───────────────────────────────────────────────────────────────

def save_grid(images, titles, path, ncols=None, figsize_per_cell=(4, 4)):
    n     = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Stage A: Sketch → Object image ───────────────────────────────────────────

def generate_from_sketch(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int, device: str,
) -> Image.Image:
    """
    Qwen does not accept FLUX LoRAs.  We pass the sketch as the input image
    and instruct Qwen to render it photorealistically via text.
    Qwen-Image-Edit LoRAs (trained on Qwen-Image) can be loaded separately
    if available; that is left as a future extension.
    """
    sketch_pil = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    prompt = (
        f"Render this sketch as a photorealistic {description} "
        "on a plain white background. No shadows. Studio lighting. High quality."
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        image=sketch_pil, prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
        num_inference_steps=num_steps, true_cfg_scale=guidance,
        height=height, width=width, generator=generator,
    ).images[0]


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "black ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)
_SEP = "═" * 60


# ── Object mask extraction ────────────────────────────────────────────────────

def _compute_obj_mask(obj_img: Image.Image,
                      grey: tuple = (128, 128, 128),
                      tolerance: int = 20) -> np.ndarray:
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


# ── Placement mask → bbox ─────────────────────────────────────────────────────

def _bbox_from_placement_mask(
    mask_path: str, width: int, height: int,
) -> Tuple[int, int, int, int]:
    mask_img = Image.open(mask_path).convert("L")
    mask_np  = np.array(mask_img)
    mh, mw   = mask_np.shape
    ys, xs   = np.where(mask_np > 127)
    if len(ys) == 0:
        return width // 6, height // 2, 5 * width // 6, height
    x1 = max(0,      int(xs.min() * width  / mw))
    y1 = max(0,      int(ys.min() * height / mh))
    x2 = min(width,  int((xs.max() + 1) * width  / mw))
    y2 = min(height, int((ys.max() + 1) * height / mh))
    return x1, y1, x2, y2


# ── Placement mask → flat token zone ─────────────────────────────────────────

def _placement_mask_to_token_zone(mask_path: str, height: int, width: int,
                                   pipe) -> np.ndarray:
    """Convert placement mask to flat bool array (n_img,) at token resolution."""
    vae_sf       = getattr(pipe, "vae_scale_factor", 8)
    h_tok, w_tok = height // (vae_sf * 2), width // (vae_sf * 2)
    mask_down    = Image.open(mask_path).convert("L").resize((w_tok, h_tok), Image.Resampling.NEAREST)
    zone = (np.array(mask_down) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_tok * w_tok} tokens ({100*zone.mean():.1f}%)")
    return zone


# ── Sobel edge extraction ─────────────────────────────────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    H, W   = img.shape[:2]
    small  = cv2.resize(img,  (256, 256))
    mask_s = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask_s, kernel, iterations=2)
    mask_s = eroded if eroded.any() else mask_s
    sx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                          cv2.convertScaleAbs(sy), 0.5, 0)
    mag = np.max(mag, axis=-1) * mask_s
    mag[mag < thresh] = 0.0
    mag3 = np.stack([mag, mag, mag], axis=-1)
    edge = (mag3.astype(np.float32) / 255.0 * small.astype(np.float32)).astype(np.uint8)
    return cv2.resize(edge, (W, H))


# ── Core collage builder ──────────────────────────────────────────────────────

def build_collage_scene(
    scene:        Image.Image,
    obj_img:      Image.Image,
    ref_mask:     np.ndarray,
    target_bbox:  Tuple[int, int, int, int],
    collage_mode: str = "full",
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    ys, xs = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using center crop fallback")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh // 4, 3 * oh // 4, ow // 4, 3 * ow // 4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop  = obj_np[oy1:oy2, ox1:ox2]
    mask_crop = ref_mask[oy1:oy2, ox1:ox2]

    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  {zone_w}×{zone_h} px")

    if collage_mode == "sobel":
        obj_z      = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z     = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        print(f"    [COLLAGE] AnyDoor hard paste {zone_w}×{zone_h} Sobel at "
              f"scene[{y1}:{y2},{x1}:{x2}]")
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    oh, ow = obj_crop.shape[:2]
    scale  = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h  = max(4, int(oh * scale))
    new_w  = max(4, int(ow * scale))

    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h))
    mask_rs = (mask_rs > 0.5).astype(np.float32)

    if collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8), mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:
        paste_layer = obj_rs

    mask_3 = np.stack([mask_rs, mask_rs, mask_rs], axis=-1)
    dy = (zone_h - new_h) // 2
    dx = (zone_w - new_w) // 2
    py1, px1 = y1 + dy, x1 + dx
    py2, px2 = py1 + new_h, px1 + new_w

    py1 = max(0, py1); py2 = min(H, py2)
    px1 = max(0, px1); px2 = min(W, px2)
    ch = py2 - py1; cw = px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region clipped to zero, using scene as-is")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]
    mask_3      = mask_3[:ch, :cw]
    collage_np  = scene_np.copy()
    scene_patch = collage_np[py1:py2, px1:px2]
    collage_np[py1:py2, px1:px2] = paste_layer * mask_3 + scene_patch * (1.0 - mask_3)
    collage_np = np.clip(collage_np, 0, 255).astype(np.uint8)

    print(f"    [COLLAGE] Pasted {new_w}×{new_h} obj at scene[{py1}:{py2}, {px1}:{px2}]  "
          f"mode={collage_mode}")
    return Image.fromarray(collage_np), (y1, y2, x1, x2)


# ── Object footprint mask (used for BCG tight boundary) ──────────────────────

def _collage_obj_mask(collage: Image.Image, scene: Image.Image,
                      threshold: int = 10) -> np.ndarray:
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask


# ── BLD latent-space BCG (Blended Latent Diffusion, arxiv:2206.02779) ─────────

def _make_bcg_latent_callback(
    scene:        Image.Image,
    mask_path:    str,
    pipe,
    height:       int,
    width:        int,
    device:       str,
    dtype:        torch.dtype = torch.bfloat16,
    dilation_lat: int = 3,
):
    """
    BLD latent BCG — identical logic to the Kontext version with two fixes:

    1. VAE normalisation read dynamically: Qwen uses the Wan2.1-based VAE
       which has different shift_factor / scaling_factor from FLUX.
    2. Latent spatial dimensions inferred from the actual VAE encode output
       so the callback works regardless of whether the pipeline returns
       unpacked (h//8) or packed (h//16) latents in callback_on_step_end.
    """
    scene_np = np.array(scene.convert("RGB"), dtype=np.float32) / 255.0
    scene_t  = torch.from_numpy(scene_np).permute(2, 0, 1).unsqueeze(0)
    scene_t  = (scene_t * 2.0 - 1.0).to(dtype)
    # Wan2.1-based VAE expects (B, C, T, H, W); add temporal dim T=1 for images
    scene_t  = scene_t.unsqueeze(2)

    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0_raw = pipe.vae.encode(scene_t.to(enc_dev)).latent_dist.sample()
        # Read shift/scale from config — Qwen's Wan2.1 VAE differs from FLUX
        sf    = getattr(pipe.vae.config, "shift_factor",   0.0)
        scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
        z0_raw = (z0_raw - sf) * scale
    # z0_raw is (1, C, 1, H_lat, W_lat); squeeze T — denoising latents are 4D
    z0 = z0_raw.squeeze(2).to(device, dtype)

    # Infer latent spatial dims from the actual encoded z0 (robust to any VAE stride)
    lat_h, lat_w = z0.shape[-2], z0.shape[-1]

    noise = torch.randn(z0.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    placement_np = np.array(
        Image.open(mask_path).convert("L").resize((lat_w, lat_h), Image.Resampling.NEAREST)
    )
    mask_lat = (placement_np > 127).astype(np.uint8)
    if dilation_lat > 0:
        k = dilation_lat * 2 + 1
        mask_lat = cv2.dilate(mask_lat, np.ones((k, k), np.uint8), iterations=2)
    mask_soft   = cv2.GaussianBlur(mask_lat.astype(np.float32), (7, 7), 2.0)
    mask_tensor = torch.from_numpy(mask_soft).to(device, dtype)[None, None]

    print(f"    [BLD] VAE latent {lat_h}×{lat_w}  "
          f"shift={sf:.4f}  scale={scale:.4f}  "
          f"edit zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape[-2:] != (lat_h, lat_w):
            # Packed latents from some schedulers — skip BCG silently
            return callback_kwargs

        # Flow-matching: timestep ≈ sigma × 1000
        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))
        z_bg  = ((1.0 - sigma) * z0 + sigma * noise).to(latents.dtype).to(latents.device)
        m     = mask_tensor.expand_as(latents).to(latents.dtype).to(latents.device)
        callback_kwargs["latents"] = latents * m + z_bg * (1.0 - m)
        return callback_kwargs

    return _callback


# ── Qwen collage K/V injection ───────────────────────────────────────────────
#
# Architecture:  hidden = [text_tokens | image_tokens]   (joint attention)
#
# Phase 1 — capture (1 forward pass, no CFG, near-clean collage):
#   K, V at target_zone image-token positions are stored per layer.
#   These encode the pasted object's appearance at the correct spatial position
#   with matching RoPE coordinates → no positional collision.
#
# Phase 2 — denoising with injection (cutoff_frac window, default late steps):
#   K[zone] = (1-s)*K + s*K_ref
#   V[zone] = (1-s)*V + s*V_ref
#   Late-step injection (sigma↓) avoids the clean/noisy mismatch that causes
#   artifacts when clean reference K/V are injected at high-noise early steps.


def _apply_complex_rope(
    x: torch.Tensor,      # (B, S, H, D)  — S-major layout, NOT yet transposed
    freqs: torch.Tensor,  # (S, D//2) complex64
) -> torch.Tensor:
    """
    Qwen/Wan2.1 complex-valued RoPE.
    freqs = (img_freqs, txt_freqs) is split by the caller; each is complex64
    with shape (S, D//2).  x must be in (B, S, H, D) layout.
    """
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (B, S, H, D//2) complex  *  (S, 1, D//2) complex  →  (B, S, H, D//2)
    x_out = torch.view_as_real(x_c * freqs.unsqueeze(1).to(x_c.device)).flatten(-2)
    return x_out.to(x.dtype)


def _qwen_attn_forward(
    attn,
    hidden_states: torch.Tensor,                    # image tokens (B, S_img, D)
    encoder_hidden_states: Optional[torch.Tensor],  # text tokens  (B, S_txt, D)
    kwargs: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[int], int, int, int]:
    """
    Correct Qwen QKV projection + complex RoPE.
    - Separate projection matrices per stream (to_q/k/v for image;
      add_q/k/v_proj for text).
    - Complex RoPE applied in (B, S, H, D) layout before transpose.
    - Joint sequence is [txt, img] (text-first, Qwen convention).
    Returns (q, k, v, txt_len, seq, B, hd) where q/k/v are (B, H, S_total, D).
    """
    B     = hidden_states.shape[0]
    S_img = hidden_states.shape[1]
    hd    = hidden_states.shape[-1] // attn.heads
    H     = attn.heads

    # ── Image stream ─────────────────────────────────────────────────────────
    img_q = attn.to_q(hidden_states).view(B, S_img, H, hd)  # (B, S_img, H, D)
    img_k = attn.to_k(hidden_states).view(B, S_img, H, hd)
    img_v = attn.to_v(hidden_states).view(B, S_img, H, hd)
    if attn.norm_q is not None: img_q = attn.norm_q(img_q)
    if attn.norm_k is not None: img_k = attn.norm_k(img_k)

    if encoder_hidden_states is not None:
        # ── Text stream ───────────────────────────────────────────────────────
        S_txt = encoder_hidden_states.shape[1]
        txt_q = attn.add_q_proj(encoder_hidden_states).view(B, S_txt, H, hd)
        txt_k = attn.add_k_proj(encoder_hidden_states).view(B, S_txt, H, hd)
        txt_v = attn.add_v_proj(encoder_hidden_states).view(B, S_txt, H, hd)
        if hasattr(attn, "norm_added_q") and attn.norm_added_q is not None:
            txt_q = attn.norm_added_q(txt_q)
        if hasattr(attn, "norm_added_k") and attn.norm_added_k is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ── Complex RoPE (separate freqs per stream) ──────────────────────────
        rot = kwargs.get("image_rotary_emb")
        if rot is not None:
            img_freqs, txt_freqs = rot
            img_q = _apply_complex_rope(img_q, img_freqs)
            img_k = _apply_complex_rope(img_k, img_freqs)
            txt_q = _apply_complex_rope(txt_q, txt_freqs)
            txt_k = _apply_complex_rope(txt_k, txt_freqs)

        # ── [txt, img] concat then transpose for SDPA ─────────────────────────
        q = torch.cat([txt_q, img_q], dim=1).transpose(1, 2)  # (B, H, S_tot, D)
        k = torch.cat([txt_k, img_k], dim=1).transpose(1, 2)
        v = torch.cat([txt_v, img_v], dim=1).transpose(1, 2)
        txt_len = S_txt
    else:
        rot = kwargs.get("image_rotary_emb")
        if rot is not None:
            img_freqs = rot[0] if isinstance(rot, (tuple, list)) else rot
            img_q = _apply_complex_rope(img_q, img_freqs)
            img_k = _apply_complex_rope(img_k, img_freqs)
        q = img_q.transpose(1, 2)
        k = img_k.transpose(1, 2)
        v = img_v.transpose(1, 2)
        txt_len = None

    seq = q.shape[2]
    return q, k, v, txt_len, seq, B, hd


def _qwen_attn_output(attn, out, txt_len, encoder_hidden_states, B, seq, heads, hd):
    """Shared output projection. Mirrors how FLUX double/single blocks work."""
    out = out.transpose(1, 2).reshape(B, seq, heads * hd).to(out.dtype)
    if encoder_hidden_states is not None:
        enc_out = out[:, :txt_len]
        img_out = out[:, txt_len:]
        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        enc_out = attn.to_add_out(enc_out) if hasattr(attn, "to_add_out") else enc_out
        return img_out, enc_out
    else:
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)
        return out


class _QwenKVCapture:
    """
    Attention processor: 1-pass capture of K/V at target zone image tokens.
    Run once on the clean collage before denoising begins.
    """
    def __init__(self, n_img: int, target_zone: np.ndarray):
        self.n_img       = n_img
        self.target_zone = target_zone  # bool (n_img,)
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer      = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        kw = {**kwargs, "image_rotary_emb": image_rotary_emb}
        q, k, v, txt_len, seq, B, hd = _qwen_attn_forward(
            attn, hidden_states, encoder_hidden_states, kw
        )
        # Image tokens are the last n_img positions in the joint sequence
        img_off  = (txt_len if txt_len is not None else seq - self.n_img)
        zone_idx = torch.from_numpy(
            img_off + np.where(self.target_zone)[0]
        ).to(k.device)

        if zone_idx.numel() > 0 and int(zone_idx.max()) < seq:
            # Store from first batch element — avoids CFG-doubled-batch confusion
            self.kv[self._layer] = (
                k[0:1, :, zone_idx, :].detach().cpu(),
                v[0:1, :, zone_idx, :].detach().cpu(),
            )

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        self._layer += 1
        return _qwen_attn_output(
            attn, out, txt_len if txt_len is not None else seq - self.n_img,
            encoder_hidden_states, B, seq, attn.heads, hd,
        )


class _QwenKVInject:
    """
    Attention processor: injects captured collage K/V at target zone image
    tokens during the active denoising window (cutoff_frac).
    """
    def __init__(self, n_img: int, target_zone: np.ndarray,
                 kv_ref: dict, obj_strength: float,
                 n_steps: int, cutoff_frac: Tuple[float, float]):
        self.n_img        = n_img
        self.target_zone  = target_zone
        self.kv_ref       = kv_ref
        self.obj_strength = obj_strength
        self.n_steps      = n_steps
        self.cutoff_frac  = cutoff_frac
        self._layer = 0
        self._step  = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        kw = {**kwargs, "image_rotary_emb": image_rotary_emb}
        q, k, v, txt_len, seq, B, hd = _qwen_attn_forward(
            attn, hidden_states, encoder_hidden_states, kw
        )
        img_off = (txt_len if txt_len is not None else seq - self.n_img)

        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)

        if lo <= self._step < hi and self._layer in self.kv_ref:
            zone_idx = torch.from_numpy(
                img_off + np.where(self.target_zone)[0]
            ).to(k.device)

            if zone_idx.numel() > 0 and int(zone_idx.max()) < seq:
                k_ref, v_ref = self.kv_ref[self._layer]
                # Expand to full batch (handles CFG-doubled batch during denoising)
                k_ref = k_ref.expand(B, -1, -1, -1).to(k.device, k.dtype)
                v_ref = v_ref.expand(B, -1, -1, -1).to(v.device, v.dtype)

                s = self.obj_strength
                k = k.clone()
                v = v.clone()
                k[:, :, zone_idx, :] = (1 - s) * k[:, :, zone_idx, :] + s * k_ref
                v[:, :, zone_idx, :] = (1 - s) * v[:, :, zone_idx, :] + s * v_ref

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        self._layer += 1
        return _qwen_attn_output(
            attn, out, img_off, encoder_hidden_states, B, seq, attn.heads, hd,
        )


@torch.no_grad()
def run_with_qwen_kv_injection(
    pipe,
    collage:      Image.Image,
    prompt:       str,
    target_zone:  np.ndarray,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    height:       int,
    width:        int,
    obj_strength: float = 0.7,
    cutoff_frac:  Tuple[float, float] = (0.5, 1.0),
    bcg_callback  = None,
) -> Image.Image:
    """
    Phase 1 — 1-step capture pass (no CFG, batch=1):
        collage → Qwen transformer → K,V at target zone image tokens per layer

    Phase 2 — full denoising with injection:
        Within cutoff_frac steps, blend captured K,V into live image tokens
        at target zone positions.  Default window is the late 50% of steps
        (sigma↓) where the latent is close to clean — matching the noise level
        of the captured K,V and avoiding early-step artifacts.
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_img  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))

    orig_procs = pipe.transformer.attn_processors

    # ── Phase 1: capture ─────────────────────────────────────────────────────
    capture_proc = _QwenKVCapture(n_img=n_img, target_zone=target_zone)
    pipe.transformer.set_attn_processor(capture_proc)

    # No negative_prompt → CFG disabled → batch stays at 1 (clean capture)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(
        image=collage, prompt=prompt,
        num_inference_steps=1, true_cfg_scale=1.0,
        height=height, width=width,
        generator=gen0, output_type="latent",
    )
    kv_ref = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(kv_ref)} / 60 layers  "
          f"zone={target_zone.sum()} tokens ({100*target_zone.mean():.1f}%)")

    # ── Phase 2: denoising with injection ─────────────────────────────────────
    inject_proc = _QwenKVInject(
        n_img=n_img, target_zone=target_zone, kv_ref=kv_ref,
        obj_strength=obj_strength, n_steps=num_steps, cutoff_frac=cutoff_frac,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pipe_ref, step_idx, timestep, cb_kwargs):
        inject_proc._step  = step_idx + 1
        inject_proc._layer = 0
        if bcg_callback is not None:
            cb_kwargs = bcg_callback(pipe_ref, step_idx, timestep, cb_kwargs)
        return cb_kwargs

    tensor_inputs = ["latents"] if bcg_callback is not None else []
    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=collage, prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
        num_inference_steps=num_steps, true_cfg_scale=guidance,
        height=height, width=width, generator=gen1,
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=tensor_inputs,
    ).images[0]

    pipe.transformer.set_attn_processor(orig_procs)
    return result


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    seed:           int,
    num_steps:      int,
    obj_guidance:   float,
    scene_guidance: float,
    collage_mode:   str,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool = False,
    obj_strength:   float = 0.7,
    cutoff_frac:    Tuple[float, float] = (0.5, 1.0),
) -> List[Image.Image]:
    results = [base]
    scene   = base

    for i, edit in enumerate(edits):
        name = edit["name"]
        desc = edit["description"]

        sketch_path = os.path.join(sketch_dir, f"{name}.png")
        if not os.path.isfile(sketch_path):
            sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found in {sketch_dir!r}. "
                f"Expected '{name}.png' or 'sketch_{name}.png'."
            )

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}")
        print(f"{'─'*60}")

        # Stage A: sketch → object image
        obj_cache = os.path.join(out_dir, f"obj_gen_{name}.png")
        if os.path.isfile(obj_cache):
            print(f"  [A] Reusing cached obj_gen_{name}.png")
            obj_img = Image.open(obj_cache).convert("RGB")
        else:
            print(f"  [A] Generating '{desc}' from sketch ...")
            obj_img = generate_from_sketch(
                pipe=pipe, sketch_path=sketch_path, description=desc,
                seed=seed, num_steps=num_steps, guidance=obj_guidance,
                height=height, width=width, device=device,
            )
            obj_img.save(obj_cache)
            print(f"      Saved: obj_gen_{name}.png")

        # Stage BBOX: placement mask required
        mask_path = os.path.join(sketch_dir, f"mask_{name}.png")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Placement mask not found: {mask_path}\n"
                f"Draw a white region on a black image to mark where '{name}' should go."
            )
        print(f"  [BBOX] Placement mask: mask_{name}.png")
        bx1, by1, bx2, by2 = _bbox_from_placement_mask(mask_path, width, height)
        print(f"      bbox: x=[{bx1},{bx2}] y=[{by1},{by2}]")
        with open(os.path.join(out_dir, f"bbox_{name}.txt"), "w") as f:
            f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

        # Stage MASK: extract object mask
        ref_mask = _compute_obj_mask(obj_img)
        n_px = ref_mask.sum()
        print(f"  [MASK] Object pixels: {n_px} ({100*n_px/(height*width):.1f}%)")
        if n_px < 50:
            print("         Fallback: using center 50% crop as object region")
            ref_mask = np.zeros((height, width), dtype=np.uint8)
            ref_mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 1

        # Stage COL: build collage
        print(f"  [COL] Building collage scene (mode={collage_mode}) ...")
        collage_scene, _ = build_collage_scene(
            scene=scene, obj_img=obj_img, ref_mask=ref_mask,
            target_bbox=(bx1, by1, bx2, by2), collage_mode=collage_mode,
        )
        collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
        print(f"      Saved collage: collage_{name}.png")

        # Object footprint mask — tight BCG boundary
        obj_mask_har = _collage_obj_mask(collage_scene, scene)
        n_har = int(obj_mask_har.sum())
        print(f"  [MASK] Object footprint: {n_har} px ({100*n_har/obj_mask_har.size:.1f}%)")

        # Stage K: Qwen integration pass with BLD latent BCG
        blend_p = f"A photorealistic room with a {desc} placed naturally."
        print(f"  [K] Qwen integration pass ...")
        print(f"      Prompt: {blend_p}")
        with open(os.path.join(out_dir, f"blend_prompt_{name}.txt"), "w") as f:
            f.write(blend_p)

        pipe_dtype = next(pipe.transformer.parameters()).dtype
        bcg_cb = _make_bcg_latent_callback(
            scene=scene, mask_path=mask_path, pipe=pipe,
            height=height, width=width, device=device, dtype=pipe_dtype,
        )

        if use_kv:
            target_zone = _placement_mask_to_token_zone(mask_path, height, width, pipe)
            print(f"      K/V zone: {target_zone.sum()} tokens  "
                  f"strength={obj_strength}  cutoff={cutoff_frac}")
            next_scene = run_with_qwen_kv_injection(
                pipe=pipe, collage=collage_scene, prompt=blend_p,
                target_zone=target_zone,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
                obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                bcg_callback=bcg_cb,
            )
        else:
            next_scene = run_qwen(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width, bcg_callback=bcg_cb,
            )

        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        # Stage BCG: pixel-space tight-mask composite
        print(f"  [BCG] Object-tight background restore ...")
        bcg_mask = cv2.dilate(obj_mask_har, np.ones((21, 21), np.uint8), iterations=3)
        bcg_mask = cv2.GaussianBlur(bcg_mask.astype(np.float32), (41, 41), 15)

        result_np = np.array(next_scene.convert("RGB"), dtype=np.float32)
        scene_np  = np.array(scene.convert("RGB"),      dtype=np.float32)
        m3        = bcg_mask[:, :, None]
        bcg_np    = np.clip(result_np * m3 + scene_np * (1.0 - m3), 0, 255).astype(np.uint8)
        next_scene = Image.fromarray(bcg_np)
        bcg_path = os.path.join(out_dir, f"result_bcg_{name}.png")
        next_scene.save(bcg_path)
        print(f"      Saved: {bcg_path}")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AnyDoor collage method inside Qwen-Image-Edit-2509."
    )
    p.add_argument("--sketch_dir",   required=True,
                   help="Directory containing {name}.png sketches and mask_{name}.png files.")
    p.add_argument("--hf_token",     required=True)
    p.add_argument("--cache_dir",    default="./models")
    p.add_argument("--out_dir",      default="results/phase1_qwen")
    p.add_argument("--model",        default="Qwen/Qwen-Image-Edit-2509",
                   help="HuggingFace model ID. Default: Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--config",       default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_guidance", type=float, default=3.5,
                   help="CFG scale for Stage A (object generation). Default 3.5.")
    p.add_argument("--guidance",      type=float, default=3.5,
                   help="CFG scale for Stage K (scene integration). Default 3.5.")
    p.add_argument("--collage_mode",  default="full",
                   choices=["full", "sobel", "blend"],
                   help="Collage paste mode. Default: full.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_steps",    type=int, default=50,
                   help="Inference steps. Qwen typically needs 50. Default 50.")
    p.add_argument("--height",       type=int, default=1024)
    p.add_argument("--width",        type=int, default=1024)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--use_kv",       action="store_true",
                   help="Enable collage K/V injection for object appearance preservation.")
    p.add_argument("--obj_strength", type=float, default=0.7,
                   help="K/V injection strength (0-1). Default 0.7.")
    p.add_argument("--cutoff_frac",  default="0.5,1.0",
                   help="Injection active window as 'start,end' fractions. Default '0.5,1.0'.")
    p.add_argument("--cpu_offload",  action="store_true",
                   help="Enable model CPU offload. Use if VRAM < 45 GB.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)

    print(f"\n{_SEP}")
    print(f"  phase1_collage_qwen  —  AnyDoor Collage + Qwen-Image-Edit-2509")
    print(f"{_SEP}")
    print(f"  Model        : {args.model}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Collage mode : {args.collage_mode}")
    print(f"  Guidance     : obj={args.lora_guidance}  scene={args.guidance}")
    print(f"  KV injection : {args.use_kv}"
          + (f"  strength={args.obj_strength}  cutoff={args.cutoff_frac}"
             if args.use_kv else ""))
    print(f"  Steps        : {args.num_steps}")
    print(f"  CPU offload  : {args.cpu_offload}")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print(f"Loading {args.model} ...")
    pipe = load_qwen_pipeline(
        model_path  = args.model,
        hf_token    = args.hf_token,
        device      = args.device,
        cache_dir   = args.cache_dir,
        cpu_offload = args.cpu_offload,
    )

    # Base scene: pass a grey canvas, ask Qwen to generate the room
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_qwen(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    _cf = [float(x) for x in args.cutoff_frac.split(",")]
    cutoff_frac: Tuple[float, float] = (_cf[0], _cf[1])

    results = run_collage_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir,
        seed=args.seed, num_steps=args.num_steps,
        obj_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        collage_mode=args.collage_mode,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_kv=args.use_kv,
        obj_strength=args.obj_strength,
        cutoff_frac=cutoff_frac,
    )

    all_imgs = results
    all_lbls = ["base"] + [e["name"] for e in edits]
    save_grid(all_imgs, all_lbls,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
