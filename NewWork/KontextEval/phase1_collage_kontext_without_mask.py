# -*- coding: utf-8 -*-
"""
phase1_collage_kontext_without_mask.py — AnyDoor Collage + FLUX Kontext, mask-free

Differences from phase1_collage_kontext.py
-------------------------------------------
  1. No user-drawn mask_{name}.png files required.
  2. Object placement is decided by monocular depth estimation (Depth Anything V2)
     — no VLM, no annotation.
  3. A rectangle mask is auto-generated from the depth-predicted bbox.

Two-image concept
-----------------
The user originally asked: "can we pass [obj_img, scene] and let the model
decide placement?"  Here this is realised at two levels:

  • Placement level: a depth model analyses the scene and selects the floor
    region where the object should go (flat surface, correct perspective).
  • Diffusion level: FLUX Kontext receives a COLLAGE image as reference —
    scene with obj_img already pasted at the depth-selected position —
    so the diffusion model literally sees both images fused into one prior.

Placement strategy (no VLM)
-----------------------------
  Insert : Depth map of scene → find flat floor region → bbox
  Remove : If object was inserted this session, reuse its tracked bbox.
           Else: depth anomaly detection (objects appear as "bumps" above
           the floor median in the lower image region).

Depth model
-----------
  Primary  : depth-anything/Depth-Anything-V2-Small-hf   (~50 MB, fast)
  Fallback : Intel/dpt-large                             (~800 MB)
  Fallback²: geometric heuristic (lower 40 % of frame)  (no download)

Pipeline
--------
  Stage A    : Sketch → LoRA FLUX → obj_img
  Stage DEPTH: Depth Anything estimates scene depth → floor bbox
  Stage MASK : Threshold obj_img → ref_mask (silhouette)
  Stage COL  : Paste obj_img at floor bbox → collage_scene
               (FLUX sees obj + scene together in one reference image)
  Stage K    : FLUX Kontext(reference=collage_scene, prompt) → result
  Stage BCG  : BLD latent background conservation
  Loop       : result → next scene

Usage
-----
  python NewWork/KontextEval/phase1_collage_kontext_without_mask.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_collage_nomask

Key flags
----------
  --depth_model   HF model id for depth estimation.
                  Default: depth-anything/Depth-Anything-V2-Small-hf
  --collage_mode  full | sobel | blend   Default: full
  --guidance      float  Kontext CFG scale. Default 2.5.
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
from diffusers import FluxKontextPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_kontext_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
    hf_token: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "./models",
    cpu_offload: bool = False,
) -> FluxKontextPipeline:
    pipe = FluxKontextPipeline.from_pretrained(
        model_path, torch_dtype=dtype, token=hf_token, cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Kontext inference ─────────────────────────────────────────────────────────

def run_standard(
    pipe, canvas: Image.Image, prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    bcg_callback=None,
) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    cb_kwargs: dict = {}
    if bcg_callback is not None:
        cb_kwargs["callback_on_step_end"] = bcg_callback
        cb_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(
        prompt=prompt, image=canvas,
        num_inference_steps=num_steps, guidance_scale=guidance,
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

_LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."


def generate_from_sketch(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int, lora_id: str, device: str,
) -> Image.Image:
    sketch_pil = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    print(f"  Loading LoRA: {lora_id}")
    pipe.load_lora_weights(lora_id)
    prompt = (
        f"{_LORA_TRIGGER} {description} on a plain white background, "
        "photorealistic, no shadows, studio lighting, high quality."
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    obj_img = pipe(
        image=sketch_pil, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width,
        max_sequence_length=512,
        generator=generator, output_type="pil",
    ).images[0]
    pipe.unload_lora_weights()
    return obj_img


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]
BASE_PROMPT = "A empty room with a wooden floor, white walls, and a window letting in natural light."
LORA_ID     = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
_SEP        = "═" * 60


# ── Object mask extraction ────────────────────────────────────────────────────

def _compute_obj_mask(obj_img: Image.Image,
                       grey: tuple = (128, 128, 128),
                       tolerance: int = 20) -> np.ndarray:
    """Binary uint8 mask (H, W): 1=object pixel, 0=background."""
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


def _neutralize_white_bg(img: Image.Image, threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    arr[white] = fill
    return Image.fromarray(arr)


def _obj_token_mask(obj_img: Image.Image, h_lat: int, w_lat: int,
                    grey: tuple = (128, 128, 128), tol: int = 10,
                    min_frac: float = 0.05) -> np.ndarray:
    """Bool mask (h_lat*w_lat,): True = object token."""
    arr = np.array(obj_img.convert("RGB").resize((w_lat, h_lat), Image.LANCZOS))
    is_grey  = (np.abs(arr[:,:,0].astype(int) - grey[0]) <= tol) & \
               (np.abs(arr[:,:,1].astype(int) - grey[1]) <= tol) & \
               (np.abs(arr[:,:,2].astype(int) - grey[2]) <= tol)
    is_white = (arr[:,:,0] >= 230) & (arr[:,:,1] >= 230) & (arr[:,:,2] >= 230)
    mask = ~(is_grey | is_white)
    if mask.mean() < min_frac:
        mask[:] = False
        mask[h_lat//4:3*h_lat//4, w_lat//4:3*w_lat//4] = True
    return mask.reshape(-1)


# ── Depth estimation — replaces user masks ────────────────────────────────────

def load_depth_model(
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    cache_dir: str = "./models",
):
    """
    Load a monocular depth estimator.

    Returns (model, processor, kind_str) so the caller can dispatch correctly.
    Falls back through: Depth-Anything-V2 → Intel/dpt-large → None (geometric).
    """
    candidates = [
        (model_id, "depth-anything"),
        # Depth Anything V2 Large (NIPS 2024): same API, ~800 MB, far better than DPT-large
        ("depth-anything/Depth-Anything-V2-Large-hf", "depth-anything"),
    ]
    for mid, kind in candidates:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc  = AutoImageProcessor.from_pretrained(mid, cache_dir=cache_dir)
            model = AutoModelForDepthEstimation.from_pretrained(mid, cache_dir=cache_dir)
            print(f"  [Depth] Loaded '{mid}'")
            return model.eval(), proc, kind
        except Exception as e:
            print(f"  [Depth] '{mid}' unavailable: {e!s:.70}")
    print("  [Depth] No depth model loaded — using geometric floor fallback.")
    return None, None, "fallback"


@torch.no_grad()
def estimate_depth(model, processor, scene: Image.Image) -> Optional[np.ndarray]:
    """
    Run depth estimation on scene.
    Returns (H, W) float32 depth map normalised to [0, 1], where larger values
    mean CLOSER to the camera (Depth Anything convention: inverted disparity).
    Returns None if model is None.
    """
    if model is None:
        return None
    inputs = processor(images=scene, return_tensors="pt")
    # Depth models are small — run on CPU to avoid interfering with FLUX on GPU
    out = model(**inputs)
    depth = out.predicted_depth[0].numpy().astype(np.float32)
    lo, hi = depth.min(), depth.max()
    return (depth - lo) / (hi - lo + 1e-8)


def _depth_to_scene_size(depth_np: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly resize depth map to (height, width)."""
    return np.array(
        Image.fromarray((depth_np * 255).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0


def find_floor_bbox(
    depth_np:   Optional[np.ndarray],
    height:     int,
    width:      int,
    floor_frac: float = 0.42,     # image rows below this are "possible floor"
    pad_frac:   float = 0.06,     # inset padding as fraction of region size
) -> Tuple[int, int, int, int]:
    """
    Select a placement bbox on the floor using depth-guided flat-region detection.

    Algorithm
    ----------
    1. Focus on lower (1 - floor_frac) of the image.
    2. Compute Sobel gradient magnitude of the depth map in that region.
       Low gradient ↔ flat surface ↔ floor.
    3. Threshold at 50th percentile → flat-surface binary mask.
    4. Find the largest connected flat region.
    5. Return its bbox (with inset padding so the object doesn't touch edges).

    Geometric fallback (no depth model): lower-center rectangle.
    """
    floor_y0 = int(height * floor_frac)

    if depth_np is None:
        print(f"    [PLACE] Geometric fallback: center-bottom region")
        return width // 5, floor_y0, 4 * width // 5, int(height * 0.90)

    depth = _depth_to_scene_size(depth_np, height, width)
    floor = depth[floor_y0:]   # (floor_rows, W)

    # Gradient magnitude — low = flat
    sx = cv2.Sobel(floor, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(floor, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sx ** 2 + sy ** 2)

    flat_mask = (grad < np.percentile(grad, 55)).astype(np.uint8)
    k = np.ones((11, 11), np.uint8)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, k)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_mask)
    if n_labels <= 1:
        # Whole floor is uniformly flat — use centre-bottom
        x1, y1 = width // 5, floor_y0 + (height - floor_y0) // 4
        x2, y2 = 4 * width // 5, int(height * 0.90)
        print(f"    [PLACE] Uniform floor — centre-bottom: x=[{x1},{x2}] y=[{y1},{y2}]")
        return x1, y1, x2, y2

    # Largest component (skip label 0 = background)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp    = (labels == largest)
    ys, xs  = np.where(comp)

    ry1 = int(ys.min()) + floor_y0
    ry2 = int(ys.max()) + floor_y0
    rx1 = int(xs.min())
    rx2 = int(xs.max())

    px = max(12, int((rx2 - rx1) * pad_frac))
    py = max(12, int((ry2 - ry1) * pad_frac))
    x1 = max(0,      rx1 + px)
    y1 = max(0,      ry1 + py)
    x2 = min(width,  rx2 - px)
    y2 = min(height, ry2 - py)

    area_frac = comp.sum() / (flat_mask.shape[0] * flat_mask.shape[1])
    print(f"    [PLACE] Floor bbox: x=[{x1},{x2}] y=[{y1},{y2}]  "
          f"flat_area={area_frac*100:.1f}%")
    return x1, y1, x2, y2


def detect_object_on_floor(
    depth_np:   Optional[np.ndarray],
    height:     int,
    width:      int,
    floor_frac: float = 0.42,
) -> Tuple[int, int, int, int]:
    """
    Locate the most prominent non-floor object in the lower image region.

    Used for removal when the insertion bbox wasn't tracked (e.g. pre-existing
    objects).  Objects appear as "bumps" — depth values closer than the local
    floor median.

    Falls back to centre-bottom if no clear anomaly is found.
    """
    floor_y0 = int(height * floor_frac)

    if depth_np is None:
        print(f"    [DETECT] Geometric fallback: centre-bottom")
        return width // 4, floor_y0, 3 * width // 4, int(height * 0.88)

    depth = _depth_to_scene_size(depth_np, height, width)
    floor = depth[floor_y0:]    # closer values are larger in Depth-Anything

    floor_med   = np.median(floor)
    floor_std   = floor.std()
    # Objects stick OUT → significantly closer (higher depth value) than floor median
    bump_thresh = floor_med + max(0.06, 0.8 * floor_std)
    obj_mask    = (floor > bump_thresh).astype(np.uint8)

    k = np.ones((9, 9), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask)
    if n_labels <= 1:
        print(f"    [DETECT] No clear depth anomaly — using centre-bottom fallback")
        return width // 4, floor_y0, 3 * width // 4, int(height * 0.88)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp    = (labels == largest)
    ys, xs  = np.where(comp)

    x1 = max(0,      int(xs.min()))
    y1 = max(0,      int(ys.min()) + floor_y0)
    x2 = min(width,  int(xs.max()))
    y2 = min(height, int(ys.max()) + floor_y0)

    print(f"    [DETECT] Depth-anomaly bbox: x=[{x1},{x2}] y=[{y1},{y2}]")
    return x1, y1, x2, y2


def _rect_mask_from_bbox(
    x1: int, y1: int, x2: int, y2: int,
    height: int, width: int,
) -> np.ndarray:
    """Rectangle uint8 mask (H×W), 255 inside bbox."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _token_zone_from_mask_np(
    mask_np: np.ndarray, height: int, width: int, pipe,
) -> np.ndarray:
    """Convert a uint8 pixel mask (H,W) to a flat bool token mask (n_gen,)."""
    vae_sf   = getattr(pipe, "vae_scale_factor", 8)
    h_lat    = height // (vae_sf * 2)
    w_lat    = width  // (vae_sf * 2)
    mask_pil = Image.fromarray(mask_np).resize((w_lat, h_lat), Image.NEAREST)
    zone     = (np.array(mask_pil) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_lat * w_lat} tokens ({100*zone.mean():.1f}%)")
    return zone


# ── K/V injection (identical to original) ────────────────────────────────────

_TIER_A         = {0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56}
_SINGLE_TXT_LEN = 512


class _KVCapture:
    def __init__(self, n_gen: int):
        self.n_gen = n_gen
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2); k = torch.cat([ek, k], dim=2); v = torch.cat([ev, v], dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb); k = apply_rotary_emb(k, image_rotary_emb)
        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                self.kv[self._layer] = (k[:,:,r_lo:r_hi,:].detach().cpu(),
                                        v[:,:,r_lo:r_hi,:].detach().cpu())
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


class _KVInject:
    def __init__(self, n_gen, obj_kv, obj_mask_np, target_zone, obj_strength, n_steps, cutoff_frac):
        self.n_gen = n_gen; self.obj_kv = obj_kv; self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone; self.obj_strength = obj_strength
        self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2); k = k.view(B,-1,attn.heads,hd).transpose(1,2); v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        lo = int(self.cutoff_frac[0]*self.n_steps); hi = int(self.cutoff_frac[1]*self.n_steps)
        if lo <= self._step < hi and self._layer in self.obj_kv:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off, img_off + self.n_gen
            k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
            k_r, v_r = self.obj_kv[self._layer]
            k_r = k_r.to(k.device, k.dtype); v_r = v_r.to(v.device, v.dtype)
            obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
            if obj_idx.numel() > 0:
                k_obj = k_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(k_gen)
                v_obj = v_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(v_gen)
            else:
                k_obj = k_r.mean(dim=2,keepdim=True).expand_as(k_gen)
                v_obj = v_r.mean(dim=2,keepdim=True).expand_as(v_gen)
            tgt = torch.from_numpy(self.target_zone).to(k.device).view(1,1,-1,1)
            s = self.obj_strength
            k_gen = torch.where(tgt,(1-s)*k_gen+s*k_obj,k_gen)
            v_gen = torch.where(tgt,(1-s)*v_gen+s*v_obj,v_gen)
            k = k.clone(); v = v.clone()
            k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


@torch.no_grad()
def run_with_kv_injection(
    pipe, canvas, prompt, obj_img, obj_mask_np, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.7, cutoff_frac=(0.0, 0.6), bcg_callback=None,
) -> Image.Image:
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    orig_procs   = pipe.transformer.attn_processors
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=_neutralize_white_bg(obj_img),
         prompt="photorealistic object, studio lighting",
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen0, output_type="latent")
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")
    inject_proc = _KVInject(n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
                             target_zone=target_zone, obj_strength=obj_strength,
                             n_steps=num_steps, cutoff_frac=cutoff_frac)
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(image=canvas, prompt=prompt,
                  num_inference_steps=num_steps, guidance_scale=guidance,
                  height=height, width=width, max_sequence_length=512,
                  generator=gen1, output_type="pil",
                  callback_on_step_end=_step_cb,
                  callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else []).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


class _CollageKVInject:
    def __init__(self, n_gen, target_zone, obj_strength, n_steps, cutoff_frac):
        self.n_gen = n_gen; self.target_zone = target_zone
        self.obj_strength = obj_strength; self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2); k = k.view(B,-1,attn.heads,hd).transpose(1,2); v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        lo = int(self.cutoff_frac[0]*self.n_steps); hi = int(self.cutoff_frac[1]*self.n_steps)
        if lo <= self._step < hi and self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off,              img_off + self.n_gen
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
                k_ref = k[:,:,r_lo:r_hi,:];         v_ref = v[:,:,r_lo:r_hi,:]
                tgt = torch.from_numpy(self.target_zone).to(k.device).view(1,1,-1,1)
                s = self.obj_strength
                k_gen = torch.where(tgt,(1.0-s)*k_gen+s*k_ref,k_gen)
                v_gen = torch.where(tgt,(1.0-s)*v_gen+s*v_ref,v_gen)
                k = k.clone(); v = v.clone()
                k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


@torch.no_grad()
def run_with_collage_kv_injection(
    pipe, canvas, prompt, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.5, cutoff_frac=(0.0, 0.6), bcg_callback=None,
) -> Image.Image:
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    orig_procs  = pipe.transformer.attn_processors
    inject_proc = _CollageKVInject(n_gen=n_gen, target_zone=target_zone,
                                   obj_strength=obj_strength, n_steps=num_steps,
                                   cutoff_frac=cutoff_frac)
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(image=canvas, prompt=prompt,
                  num_inference_steps=num_steps, guidance_scale=guidance,
                  height=height, width=width, max_sequence_length=512,
                  generator=gen, output_type="pil",
                  callback_on_step_end=_step_cb,
                  callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else []).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


# ── Sobel edge extraction ─────────────────────────────────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    H, W    = img.shape[:2]
    small   = cv2.resize(img, (256, 256))
    mask_s  = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    eroded  = cv2.erode(mask_s, np.ones((5, 5), np.uint8), iterations=2)
    mask_s  = eroded if eroded.any() else mask_s
    sx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0)
    mag = np.max(mag, axis=-1) * mask_s; mag[mag < thresh] = 0.0
    mag3 = np.stack([mag, mag, mag], axis=-1)
    edge = (mag3.astype(np.float32) / 255.0 * small.astype(np.float32)).astype(np.uint8)
    return cv2.resize(edge, (W, H))


# ── Collage builder ───────────────────────────────────────────────────────────

def build_collage_scene(
    scene:        Image.Image,
    obj_img:      Image.Image,
    ref_mask:     np.ndarray,
    target_bbox:  Tuple[int, int, int, int],
    collage_mode: str = "full",
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Paste obj_img into scene at target_bbox.

    This is where the two-image fusion happens at the diffusion level:
    FLUX Kontext receives the collage (obj already placed in scene) as its
    reference image, so it sees both obj_img and scene in a single prior.
    """
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    ys, xs = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using centre crop")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh//4, 3*oh//4, ow//4, 3*ow//4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop  = obj_np[oy1:oy2, ox1:ox2]
    mask_crop = ref_mask[oy1:oy2, ox1:ox2]

    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  {zone_w}×{zone_h} px")

    if collage_mode == "sobel":
        obj_z  = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    oh, ow  = obj_crop.shape[:2]
    scale   = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h   = max(4, int(oh * scale))
    new_w   = max(4, int(ow * scale))
    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = (cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h)) > 0.5).astype(np.float32)

    if collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8), mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:
        paste_layer = obj_rs

    mask_3 = np.stack([mask_rs] * 3, axis=-1)
    dy = (zone_h - new_h) // 2
    dx = (zone_w - new_w) // 2
    py1, px1 = max(0, y1+dy), max(0, x1+dx)
    py2, px2 = min(H, py1+new_h), min(W, px1+new_w)
    ch, cw = py2 - py1, px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region zero — returning scene unchanged")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]; mask_3 = mask_3[:ch, :cw]
    collage_np  = scene_np.copy()
    collage_np[py1:py2, px1:px2] = paste_layer * mask_3 + collage_np[py1:py2, px1:px2] * (1.0 - mask_3)
    collage_np  = np.clip(collage_np, 0, 255).astype(np.uint8)
    print(f"    [COLLAGE] Pasted {new_w}×{new_h} at [{py1}:{py2},{px1}:{px2}]  mode={collage_mode}")
    return Image.fromarray(collage_np), (y1, y2, x1, x2)


def build_removal_collage(
    scene:     Image.Image,
    mask_np:   np.ndarray,
    height:    int,
    width:     int,
    dilate_px: int = 8,
) -> Image.Image:
    """Inpaint the object region (supplied as uint8 mask array) using Telea."""
    scene_np     = np.array(scene.convert("RGB").resize((width, height), Image.LANCZOS))
    inpaint_mask = (mask_np > 127).astype(np.uint8)
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((k, k), np.uint8), iterations=1)
    inpainted = cv2.inpaint(scene_np, inpaint_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    print(f"    [REMOVAL] Telea inpaint: {int(inpaint_mask.sum())} px erased")
    return Image.fromarray(inpainted)


def _collage_obj_mask(collage: Image.Image, scene: Image.Image,
                       threshold: int = 10) -> np.ndarray:
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask


# ── BLD latent BCG (takes mask_np, not mask_path) ────────────────────────────

def _make_bcg_latent_callback(
    scene:        Image.Image,
    mask_np:      np.ndarray,      # uint8 (H,W) — 255 inside edit zone
    pipe,
    height:       int,
    width:        int,
    device:       str,
    dtype:        torch.dtype = torch.bfloat16,
    dilation_lat: int = 3,
):
    """
    At every denoising step with noise level sigma_t:
        z_bg_t = (1 - sigma_t) * z0_scene + sigma_t * noise
        latents = latents * mask + z_bg_t * (1 - mask)
    mask_np is the auto-generated rectangle mask at pixel resolution.
    """
    vae_h = height // 8; vae_w = width // 8

    scene_np = np.array(scene.convert("RGB"), dtype=np.float32) / 255.0
    scene_t  = torch.from_numpy(scene_np).permute(2, 0, 1).unsqueeze(0)
    scene_t  = (scene_t * 2.0 - 1.0).to(dtype)
    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0 = pipe.vae.encode(scene_t.to(enc_dev)).latent_dist.sample()
        z0 = (z0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z0    = z0.to(device, dtype)
    noise = torch.randn(z0.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    mask_lat = np.array(
        Image.fromarray(mask_np).resize((vae_w, vae_h), Image.NEAREST)
    )
    mask_lat = (mask_lat > 127).astype(np.uint8)
    if dilation_lat > 0:
        k = dilation_lat * 2 + 1
        mask_lat = cv2.dilate(mask_lat, np.ones((k, k), np.uint8), iterations=2)
    mask_soft   = cv2.GaussianBlur(mask_lat.astype(np.float32), (7, 7), 2.0)
    mask_tensor = torch.from_numpy(mask_soft).to(device, dtype)[None, None]
    print(f"    [BLD] Latent mask {vae_h}×{vae_w}, edit zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape[-2:] != (vae_h, vae_w):
            return callback_kwargs
        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))
        z_bg  = ((1.0-sigma)*z0 + sigma*noise).to(latents.dtype).to(latents.device)
        m     = mask_tensor.expand_as(latents).to(latents.dtype).to(latents.device)
        callback_kwargs["latents"] = latents * m + z_bg * (1.0 - m)
        return callback_kwargs

    return _callback


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    depth_model,
    depth_processor,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    collage_mode:   str,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool  = False,
    obj_strength:   float = 0.7,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.6),
) -> List[Image.Image]:
    """
    Mask-free incremental object insertion / removal.

    Per insert:
      Stage A    : Sketch → LoRA FLUX → obj_img
      Stage DEPTH: estimate_depth(scene) → find_floor_bbox() → mask_np
      Stage MASK : obj silhouette from obj_img
      Stage COL  : paste obj_img at floor bbox → collage_scene   (two-image fusion)
      Stage K    : FLUX Kontext(collage_scene, prompt) + BLD BCG → result

    Per remove:
      Stage DEPTH: use tracked insertion bbox if available,
                   else detect_object_on_floor(scene depth) → mask_np
      Stage COL  : Telea inpaint the mask region → removal collage
      Stage K    : FLUX Kontext(removal_collage, prompt) + BLD BCG → result

    Insertion bboxes are tracked in _placed dict so removal can reuse them
    without needing a second detection step.
    """
    results = [base]
    scene   = base
    _placed: Dict[str, Tuple[int, int, int, int]] = {}   # track insertion bboxes

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        # ── Stage DEPTH: estimate scene depth ────────────────────────────────
        print(f"  [DEPTH] Estimating scene depth ...")
        depth_np = estimate_depth(depth_model, depth_processor, scene)
        if depth_np is not None:
            # Save depth visualisation for inspection
            depth_vis = (depth_np * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(
                os.path.join(out_dir, f"depth_{i+1}_{name}_{action}.png")
            )

        if action == "insert":
            # Stage A: sketch → object image
            sketch_path = os.path.join(sketch_dir, f"{name}.png")
            if not os.path.isfile(sketch_path):
                sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
            if not os.path.isfile(sketch_path):
                raise FileNotFoundError(
                    f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                    f"in {sketch_dir!r}"
                )
            print(f"  [A] Generating '{desc}' from sketch ...")
            obj_img = generate_from_sketch(
                pipe=pipe, sketch_path=sketch_path, description=desc,
                seed=seed, num_steps=num_steps, guidance=lora_guidance,
                height=height, width=width, lora_id=lora_id, device=device,
            )
            obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))

            # Depth-guided floor placement (no user mask needed)
            print(f"  [PLACE] Finding floor placement ...")
            bx1, by1, bx2, by2 = find_floor_bbox(depth_np, height, width)
            _placed[name] = (bx1, by1, bx2, by2)    # track for potential removal

            mask_np = _rect_mask_from_bbox(bx1, by1, bx2, by2, height, width)
            Image.fromarray(mask_np).save(
                os.path.join(out_dir, f"mask_pred_{name}.png")
            )
            with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
                f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

            # Stage MASK: object silhouette from obj_img
            ref_mask = _compute_obj_mask(obj_img)
            n_px = ref_mask.sum()
            print(f"  [MASK] Object pixels: {n_px} ({100*n_px/(height*width):.1f}%)")
            if n_px < 50:
                print("         Fallback: using centre 50% crop")
                ref_mask = np.zeros((height, width), dtype=np.uint8)
                ref_mask[height//4:3*height//4, width//4:3*width//4] = 1

            # Stage COL: paste obj into scene → FLUX sees both images merged
            print(f"  [COL] Building collage scene (mode={collage_mode}) ...")
            collage_scene, _ = build_collage_scene(
                scene=scene, obj_img=obj_img, ref_mask=ref_mask,
                target_bbox=(bx1, by1, bx2, by2), collage_mode=collage_mode,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))

            obj_mask_har = _collage_obj_mask(collage_scene, scene)
            blend_p = f"A photorealistic room with a {desc} placed naturally."

        else:  # remove
            # Use tracked insertion bbox if available (most reliable)
            if name in _placed:
                bx1, by1, bx2, by2 = _placed[name]
                print(f"  [PLACE] Reusing tracked insertion bbox: "
                      f"x=[{bx1},{bx2}] y=[{by1},{by2}]")
            else:
                # Depth-anomaly detection for pre-existing objects
                print(f"  [DETECT] Depth-anomaly object detection ...")
                bx1, by1, bx2, by2 = detect_object_on_floor(depth_np, height, width)

            mask_np = _rect_mask_from_bbox(bx1, by1, bx2, by2, height, width)
            Image.fromarray(mask_np).save(
                os.path.join(out_dir, f"mask_pred_remove_{name}.png")
            )
            with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
                f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

            print(f"  [COL] Building removal collage (Telea inpaint) ...")
            collage_scene = build_removal_collage(
                scene=scene, mask_np=mask_np, height=height, width=width,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_remove_{name}.png"))

            obj_mask_har = (mask_np > 127).astype(np.uint8)
            blend_p = (
                f"{BASE_PROMPT} "
                f"The {desc} has been removed. "
                f"Seamless wooden floor and white walls fill the area. "
                f"No object, clean empty room."
            )
            # Remove from tracking once erased
            _placed.pop(name, None)

        # ── Stage K: Kontext integration + BLD background conservation ───────
        step_tag = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"
        print(f"  [K] Kontext integration pass  kv={use_kv} ...")
        with open(os.path.join(out_dir, f"prompt_{step_tag}.txt"), "w") as f:
            f.write(blend_p)

        pipe_dtype = next(pipe.transformer.parameters()).dtype
        bcg_cb = _make_bcg_latent_callback(
            scene=scene, mask_np=mask_np, pipe=pipe,
            height=height, width=width, device=device, dtype=pipe_dtype,
        )

        if use_kv:
            target_zone = _token_zone_from_mask_np(mask_np, height, width, pipe)
            if action == "insert":
                vae_sf = getattr(pipe, "vae_scale_factor", 8)
                obj_mask_tok = _obj_token_mask(
                    obj_img,
                    height // (vae_sf * 2), width // (vae_sf * 2),
                )
                next_scene = run_with_kv_injection(
                    pipe=pipe, canvas=collage_scene, prompt=blend_p,
                    obj_img=obj_img, obj_mask_np=obj_mask_tok,
                    target_zone=target_zone,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                    bcg_callback=bcg_cb,
                )
            else:
                next_scene = run_with_collage_kv_injection(
                    pipe=pipe, canvas=collage_scene, prompt=blend_p,
                    target_zone=target_zone,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                    bcg_callback=bcg_cb,
                )
        else:
            next_scene = run_standard(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
                bcg_callback=bcg_cb,
            )

        result_path = os.path.join(out_dir, f"result_{step_tag}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        # ── Stage BCG: pixel-space background restoration ────────────────────
        print(f"  [BCG] Background restore ...")
        bcg_mask = cv2.dilate(obj_mask_har, np.ones((21, 21), np.uint8), iterations=3)
        bcg_mask = cv2.GaussianBlur(bcg_mask.astype(np.float32), (41, 41), 15)
        result_np = np.array(next_scene.convert("RGB"), dtype=np.float32)
        scene_np  = np.array(scene.convert("RGB"),      dtype=np.float32)
        m3        = bcg_mask[:, :, None]
        bcg_np    = np.clip(result_np * m3 + scene_np * (1.0 - m3), 0, 255).astype(np.uint8)
        next_scene = Image.fromarray(bcg_np)
        bcg_path   = os.path.join(out_dir, f"result_bcg_{step_tag}.png")
        next_scene.save(bcg_path)
        print(f"      Saved: {bcg_path}")

        scene = next_scene
        results.append(scene)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Two-image chain: let FLUX decide placement ────────────────────────────────

def run_two_image_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool  = False,
    obj_strength:   float = 0.7,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.6),
) -> List[Image.Image]:
    """
    Two-image Kontext chain — no depth, no collage, no mask.

    Reference = vertical composite: obj (top, full res) + scene (bottom, full res).
    FLUX decides entirely where to place the object.

    With --kv_injection:
      A 1-step capture pass records K/V from obj_img at TIER_A layers.
      During composite denoising those K/V are injected into ALL gen tokens
      (full-field, no zone restriction) so FLUX retains obj appearance wherever
      it decides to place it.

    Outputs saved as two_image_step{N}_{name}.png for comparison.
    """
    results = [base]
    scene   = base

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")

        print(f"\n{'─'*60}")
        print(f"  [TWO-IMAGE] Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        generator = torch.Generator(device=pipe.device).manual_seed(seed)

        if action == "insert":
            # Stage A: sketch → object image
            sketch_path = os.path.join(sketch_dir, f"{name}.png")
            if not os.path.isfile(sketch_path):
                sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
            if not os.path.isfile(sketch_path):
                raise FileNotFoundError(
                    f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                    f"in {sketch_dir!r}"
                )
            print(f"  [A] Generating '{desc}' from sketch ...")
            obj_img = generate_from_sketch(
                pipe=pipe, sketch_path=sketch_path, description=desc,
                seed=seed, num_steps=num_steps, guidance=lora_guidance,
                height=height, width=width, lora_id=lora_id, device=device,
            )
            obj_img.save(os.path.join(out_dir, f"two_image_obj_{name}.png"))

            # Stack obj on top of scene at full resolution — no squeezing.
            # height/width in the pipe() call set the OUTPUT size; the reference
            # image is just encoded to latents independently so any size works.
            obj_full  = obj_img.resize((width, height), Image.LANCZOS)
            composite = Image.new("RGB", (width, height * 2))
            composite.paste(obj_full, (0, 0))
            composite.paste(scene,    (0, height))
            composite.save(os.path.join(out_dir, f"two_image_composite_{name}.png"))

            prompt = (
                f"The top image shows a {desc} on a white background. "
                f"The bottom image shows an empty room scene. "
                f"Insert the {desc} from the top image naturally into the room. "
                f"Place it on the floor with correct perspective, lighting, and scale. "
                f"Add a contact shadow. Output the complete room scene."
            )

            if use_kv:
                # Full-field K/V injection: target_zone = all gen tokens.
                # No placement mask exists, so we reinforce obj appearance
                # everywhere and let FLUX decide the location.
                vae_sf  = getattr(pipe, "vae_scale_factor", 8)
                h_lat   = height // (vae_sf * 2)
                w_lat   = width  // (vae_sf * 2)
                n_gen   = h_lat * w_lat
                full_zone    = np.ones(n_gen, dtype=bool)
                obj_mask_tok = _obj_token_mask(obj_img, h_lat, w_lat)
                print(f"  [K] KV-injection (full-field) + composite reference ...")
                next_scene = run_with_kv_injection(
                    pipe=pipe, canvas=composite, prompt=prompt,
                    obj_img=obj_img, obj_mask_np=obj_mask_tok,
                    target_zone=full_zone,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                )
            else:
                print(f"  [K] pipe(composite) — model decides placement ...")
                next_scene = pipe(
                    prompt=prompt,
                    image=composite,
                    num_inference_steps=num_steps,
                    guidance_scale=scene_guidance,
                    height=height, width=width,
                    generator=generator,
                ).images[0]

        else:  # remove
            prompt = (
                f"{BASE_PROMPT} "
                f"Remove the {desc} completely. "
                f"Fill the area with seamless wooden floor and white walls. "
                f"No object, clean empty room."
            )
            print(f"  [K] pipe(image=[scene]) — removal via prompt only ...")
            next_scene = pipe(
                prompt=prompt,
                image=scene,
                num_inference_steps=num_steps,
                guidance_scale=scene_guidance,
                height=height, width=width,
                generator=generator,
            ).images[0]

        step_tag = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"
        out_path = os.path.join(out_dir, f"two_image_{step_tag}.png")
        next_scene.save(out_path)
        print(f"      Saved: {out_path}")

        with open(os.path.join(out_dir, f"prompt_two_image_{step_tag}.txt"), "w") as f:
            f.write(prompt)

        scene = next_scene
        results.append(scene)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AnyDoor collage + FLUX Kontext — mask-free via depth estimation."
    )
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/phase1_collage_nomask")
    p.add_argument("--mode",          default="collage",
                   choices=["collage", "two_image"],
                   help="'collage': depth-guided paste then Kontext (default). "
                        "'two_image': pass [obj_img, scene] directly to FLUX and "
                        "observe where the model places the object.")
    p.add_argument("--depth_model",
                   default="depth-anything/Depth-Anything-V2-Small-hf",
                   help="HF model id for monocular depth estimation. "
                        "Default: Depth-Anything-V2-Small (~50 MB). "
                        "Falls back to Intel/dpt-large, then geometric heuristic.")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description, action}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5)
    p.add_argument("--collage_mode",  default="full",
                   choices=["full", "sobel", "blend"])
    p.add_argument("--kv_injection",  action="store_true")
    p.add_argument("--obj_strength",  type=float, default=0.7)
    p.add_argument("--cutoff_frac",   default="0.0,0.6")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
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
    print(f"  phase1_collage_kontext_without_mask  —  mode={args.mode}")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Mode         : {args.mode}")
    if args.mode == "collage":
        print(f"  Depth model  : {args.depth_model}")
        print(f"  Collage mode : {args.collage_mode}")
        print(f"  KV injection : {args.kv_injection}"
              + (f"  strength={args.obj_strength}  cutoff={args.cutoff_frac}"
                 if args.kv_injection else ""))
    print(f"  Guidance     : {args.guidance}")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    # Depth model only needed for collage mode
    if args.mode == "collage":
        print("Loading depth model ...")
        depth_model, depth_processor, _ = load_depth_model(
            model_id=args.depth_model, cache_dir=args.cache_dir,
        )
    else:
        depth_model = depth_processor = None
        print("[two_image mode] Skipping depth model — FLUX decides placement.")

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    _cf = [float(x) for x in args.cutoff_frac.split(",")]
    cutoff_frac: Tuple[float, float] = (_cf[0], _cf[1])

    if args.mode == "two_image":
        results = run_two_image_chain(
            pipe=pipe, base=base, edits=edits,
            sketch_dir=args.sketch_dir, lora_id=args.lora_id,
            seed=args.seed, num_steps=args.num_steps,
            lora_guidance=args.lora_guidance,
            scene_guidance=args.guidance,
            height=args.height, width=args.width,
            out_dir=args.out_dir, device=args.device,
            use_kv=args.kv_injection,
            obj_strength=args.obj_strength,
            cutoff_frac=cutoff_frac,
        )
    else:
        results = run_collage_chain(
            pipe=pipe, base=base, edits=edits,
            sketch_dir=args.sketch_dir, lora_id=args.lora_id,
            depth_model=depth_model, depth_processor=depth_processor,
            seed=args.seed, num_steps=args.num_steps,
            lora_guidance=args.lora_guidance,
            scene_guidance=args.guidance,
            collage_mode=args.collage_mode,
            height=args.height, width=args.width,
            out_dir=args.out_dir, device=args.device,
            use_kv=args.kv_injection,
            obj_strength=args.obj_strength,
            cutoff_frac=cutoff_frac,
        )

    all_imgs = results
    all_lbls = ["base"] + [f"{e['name']} ({e.get('action','insert')})" for e in edits]
    save_grid(all_imgs, all_lbls,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
