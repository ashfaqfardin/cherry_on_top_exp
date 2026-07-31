# -*- coding: utf-8 -*-
"""
phase1_collage_kontext.py — AnyDoor's Collage Method inside FLUX Kontext

How AnyDoor works (the idea we're stealing)
--------------------------------------------
AnyDoor builds a "hint collage": it takes the reference object, extracts its
Sobel edge map (structure without colour), then PASTES that into the target scene
at the desired location.  This collage is fed to a ControlNet so the model
SEES the object's shape at the right place and generates natural integration.

How we implement it in Kontext
-------------------------------
FLUX Kontext is conditioned on a reference image — that IS our "ControlNet".
So instead of a raw scene, we pass a COLLAGE SCENE as the Kontext reference:

  collage_scene = scene with obj pasted (soft-masked) at the target location

The VLM tells us WHERE.  FLUX then:
  1. Sees the obj shape/colour at the correct position in the reference
  2. Denoises from pure noise conditioned on that reference
  3. Naturally integrates lighting, shadows, perspective

Key differences from all previous attempts
-------------------------------------------
  phase1_sketch_vlm  : Kontext reference = raw scene, text describes obj → identity lost
  phase1_obj_kv      : K/V injection in attention space → too soft
  phase1_sdedit      : Noisy-obj init → wrong position, grey bg contamination
  phase1_roi         : Additive latent signal → FLUX doesn't respond strongly enough
  THIS FILE          : Kontext reference = scene WITH OBJ ALREADY VISIBLE at target
                       → FLUX has the full visual prior: shape, colour, position

Collage modes (--collage_mode)
-------------------------------
  'full'   (default) : paste actual obj pixels soft-masked into scene
                       Best colour/texture fidelity.
  'sobel'            : paste Sobel edge map of obj instead of pixels
                       More adaptation freedom; matches AnyDoor exactly.
  'blend'            : weighted mix: alpha*full + (1-alpha)*sobel
                       Balance between identity lock and natural adaptation.

Pipeline
---------
  Stage A   : Sketch → LoRA FLUX → obj_img   (same as all other phases)
  Stage MASK: Threshold obj_img → ref_mask    (non-white/grey pixels)
  Stage VLM : VLM(scene, obj_img) → placement description
  Stage COL : Build collage_scene by pasting obj at placement bbox
  Stage K   : FLUX Kontext(reference=collage_scene, prompt=blend_prompt) → result
  Loop      : result → next scene

Usage
-----
  python NewWork/KontextEval/phase1_collage_kontext.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_collage \\
      --vlm_model Qwen/Qwen2.5-VL-7B-Instruct

Key flags
----------
  --collage_mode   full | sobel | blend   Default: full
  --paste_alpha    float  Opacity of pasted obj (0-1). Default 0.85.
  --feather        int    Gaussian feather radius for soft mask edge. Default 25.
  --guidance       float  Kontext CFG scale. Default 2.5.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxKontextPipeline
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
) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt, image=canvas,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, generator=generator,
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


# ── VLM loading ───────────────────────────────────────────────────────────────

def load_vlm(model_id: str, cache_dir: str, device: str = "cpu"):
    """
    Load a Qwen2-VL or Qwen2.5-VL model for placement bbox prediction.

    device="cpu"  → bfloat16, safe alongside FLUX on any GPU.
    device="cuda" → tries 4-bit NF4 (bitsandbytes) first, then bf16 fallback.
                    pip install -U bitsandbytes>=0.46.1  to enable 4-bit.
    """
    from transformers import AutoProcessor

    print(f"  Loading VLM '{model_id}' on {device} ...")

    def _load_model(load_kwargs: dict, to_device: bool):
        is_qwen25 = "Qwen2.5" in model_id or "Qwen2_5" in model_id
        classes = []
        if is_qwen25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        try:
            from transformers import Qwen2VLForConditionalGeneration
            classes.append(Qwen2VLForConditionalGeneration)
        except ImportError:
            pass
        if not is_qwen25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        last_err = None
        for cls in classes:
            try:
                m = cls.from_pretrained(model_id, **load_kwargs)
                if to_device:
                    m = m.to(device)
                return m.eval()
            except Exception as e:
                last_err = e
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
            if to_device:
                m = m.to(device)
            return m.eval()
        except Exception as e:
            last_err = e
        raise RuntimeError(f"Could not load VLM '{model_id}'. Last error: {last_err}")

    model = None
    mode  = ""
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            model = _load_model(
                dict(quantization_config=quant_cfg, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            mode = "4-bit NF4 GPU"
        except Exception as e:
            print(f"    [VLM] 4-bit load failed ({e!s:.80})")
            print(f"    [VLM] Falling back to bf16 GPU")
        if model is None:
            model = _load_model(
                dict(torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            mode = "bf16 GPU"
    else:
        model = _load_model(
            dict(torch_dtype=torch.bfloat16, cache_dir=cache_dir),
            to_device=True,
        )
        mode = "bf16 CPU"

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    print(f"  VLM loaded  ({mode})")
    return model, processor

# ── constants ─────────────────────────────────────────────────────────────────
EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "white ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT = "A empty room with a wooden floor, white walls, and a window letting in natural light."
LORA_ID     = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
_SEP        = "═" * 60


# ── Object mask extraction (same logic as phase1_obj_kv) ─────────────────────

def _compute_obj_mask(obj_img: Image.Image,
                       grey: tuple = (128, 128, 128),
                       tolerance: int = 20) -> np.ndarray:
    """
    Binary uint8 mask (H, W): 1 = object pixel, 0 = background.
    Detects both white (LoRA output) and grey (neutralised) backgrounds.
    """
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


# ── Placement mask → bbox ────────────────────────────────────────────────────

def _bbox_from_placement_mask(
    mask_path: str,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Extract the bounding box of the white region in a user-drawn placement mask.
    White pixels (>127) mark where the object should be placed.
    Mask resolution is rescaled to (width × height) pixel space.
    """
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


# ── VLM bbox fallback (used when no placement mask is found) ─────────────────

def _vlm_bbox(
    vlm_model,
    vlm_processor,
    scene_img:   Image.Image,
    obj_img:     Image.Image,
    description: str,
    width:       int,
    height:      int,
    max_new_tokens: int = 64,
) -> Tuple[int, int, int, int]:
    """
    Ask the VLM only WHERE to place `description` — returns a pixel bbox.
    No text prompt generation; the collage already encodes appearance + position
    visually, so Kontext gets a fixed integration template instead.

    Parses two output formats the model may produce:
      - Native grounding tokens: <|box_start|>(x,y),(x,y)<|box_end|>  [0,1000)
      - Plain bracket format:    BBOX: [x1, y1, x2, y2]  (resized image pixels)
    Both are rescaled to full (width × height) pixel space.
    Falls back to center-floor zone if neither format is found.
    """
    import re

    device = next(vlm_model.parameters()).device

    def _resize(img, max_side):
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return img.convert("RGB")

    scene_small = _resize(scene_img, 768)
    obj_small   = _resize(obj_img,   512)
    sw, sh      = scene_small.size

    instruction = (
        f"Image 1 is a room scene. Image 2 is a {description} on a white background.\n\n"
        f"Find the best empty location in Image 1 to place the {description} "
        f"so it sits naturally on a surface (floor, table, or shelf) without "
        f"overlapping existing furniture.\n\n"
        f"Output ONLY the bounding box of that location in Image 1. Nothing else."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": scene_small},
                {"type": "image", "image": obj_small},
                {"type": "text",  "text": instruction},
            ],
        }
    ]

    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = vlm_processor(
        text=[text],
        images=[scene_small, obj_small],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    out_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
    raw = vlm_processor.batch_decode(
        out_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print(f"    [VLM-bbox] raw: {raw!r}")

    def _rescale(rx1, ry1, rx2, ry2):
        x1 = max(0,      int(rx1 * width  / sw))
        y1 = max(0,      int(ry1 * height / sh))
        x2 = min(width,  int(rx2 * width  / sw))
        y2 = min(height, int(ry2 * height / sh))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    # Parser 1: native grounding tokens — coords normalised [0, 1000)
    bm = re.search(
        r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>', raw
    )
    if bm:
        x1n, y1n, x2n, y2n = (int(bm.group(i)) for i in range(1, 5))
        bbox = _rescale(
            int(x1n * sw / 1000), int(y1n * sh / 1000),
            int(x2n * sw / 1000), int(y2n * sh / 1000),
        )
        if bbox:
            print(f"    [VLM-bbox] grounding: x=[{bbox[0]},{bbox[2]}] y=[{bbox[1]},{bbox[3]}]")
            return bbox

    # Parser 2: bracket format — floats or ints
    # Model may output normalized [0,1] coords or pixel coords in resized space.
    NUM = r'(\d+\.?\d*)'
    SEP = r'[,\s]+'
    bm2 = re.search(rf'\[{NUM}{SEP}{NUM}{SEP}{NUM}{SEP}{NUM}\]', raw)
    if bm2:
        v = [float(bm2.group(i)) for i in range(1, 5)]
        rx1, ry1, rx2, ry2 = v
        if max(v) <= 1.0:
            # normalized [0,1] → multiply directly by full image dimensions
            x1 = max(0,      int(rx1 * width))
            y1 = max(0,      int(ry1 * height))
            x2 = min(width,  int(rx2 * width))
            y2 = min(height, int(ry2 * height))
            bbox = (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
        else:
            # pixel coords in resized image space → rescale
            bbox = _rescale(rx1, ry1, rx2, ry2)
        if bbox:
            print(f"    [VLM-bbox] bracket:   x=[{bbox[0]},{bbox[2]}] y=[{bbox[1]},{bbox[3]}]")
            return bbox

    print(f"    [VLM-bbox] parse failed — center-floor fallback")
    return width // 6, height // 2, 5 * width // 6, height


# ── Sobel edge extraction (mirrors AnyDoor's sobel()) ────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    """
    Returns Sobel-filtered RGB image: high-frequency edges where the object is,
    black elsewhere.  Matches AnyDoor's detail conditioning approach.
    """
    H, W = img.shape[:2]
    small   = cv2.resize(img,  (256, 256))
    mask_s  = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    kernel  = np.ones((5, 5), np.uint8)
    mask_s  = cv2.erode(mask_s, kernel, iterations=2)

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
    collage_mode: str  = "full",
    paste_alpha:  float = 0.85,
    feather:      int   = 25,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    AnyDoor-style: build a collage scene with the object pasted at the
    target location.  Returns (collage_pil, (x1, y1, x2, y2) paste bbox).

    target_bbox   — (x1, y1, x2, y2) from _vlm_prompt_and_bbox.
    collage_mode:
      'full'  — paste actual obj pixels (colour + texture)
      'sobel' — paste Sobel edge map only (structure, matches AnyDoor exactly)
      'blend' — 60% full + 40% sobel

    The pasted region has a Gaussian-feathered soft mask so edges are smooth.
    """
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    # Crop to tight bbox around the object in obj_img
    ys, xs   = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using center crop fallback")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh//4, 3*oh//4, ow//4, 3*ow//4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop   = obj_np[oy1:oy2, ox1:ox2]           # (oh, ow, 3)
    mask_crop  = ref_mask[oy1:oy2, ox1:ox2]         # (oh, ow) uint8

    # Target placement bbox — directly from VLM grounding
    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  "
          f"{zone_w}×{zone_h} px")

    # Scale obj to fill ~80% of the target zone (preserve aspect ratio)
    oh, ow  = obj_crop.shape[:2]
    scale   = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h   = max(4, int(oh * scale))
    new_w   = max(4, int(ow * scale))

    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h))
    mask_rs = (mask_rs > 0.5).astype(np.float32)

    # Choose what to paste
    if collage_mode == "sobel":
        paste_layer = _sobel_map(obj_rs.astype(np.uint8),
                                  mask_rs).astype(np.float32)
    elif collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8),
                                  mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:  # 'full'
        paste_layer = obj_rs

    # Gaussian-feathered soft mask: smooth edges like a natural composite
    mask_f  = (mask_rs * 255).astype(np.uint8)
    mask_f  = cv2.GaussianBlur(mask_f, (feather * 2 + 1, feather * 2 + 1), feather / 3)
    mask_f  = mask_f.astype(np.float32) / 255.0 * paste_alpha
    mask_3  = np.stack([mask_f, mask_f, mask_f], axis=-1)

    # Center the resized obj in the target zone
    dy = (zone_h - new_h) // 2
    dx = (zone_w - new_w) // 2
    py1, px1 = y1 + dy, x1 + dx
    py2, px2 = py1 + new_h, px1 + new_w

    # Clamp to scene bounds
    py1 = max(0, py1);  py2 = min(H, py2)
    px1 = max(0, px1);  px2 = min(W, px2)
    ch  = py2 - py1;    cw  = px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region clipped to zero, using scene as-is")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]
    mask_3      = mask_3[:ch, :cw]

    # Alpha-composite onto scene copy
    collage_np  = scene_np.copy()
    scene_patch = collage_np[py1:py2, px1:px2]
    collage_np[py1:py2, px1:px2] = (
        paste_layer * mask_3 + scene_patch * (1.0 - mask_3)
    )
    collage_np = np.clip(collage_np, 0, 255).astype(np.uint8)

    print(f"    [COLLAGE] Pasted {new_w}×{new_h} obj at scene[{py1}:{py2}, {px1}:{px2}]  "
          f"mode={collage_mode}  alpha={paste_alpha}")

    return Image.fromarray(collage_np), (y1, y2, x1, x2)


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    vlm_pair:       tuple,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    collage_mode:   str,
    paste_alpha:    float,
    feather:        int,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
) -> List[Image.Image]:
    """
    AnyDoor-inspired incremental object insertion via FLUX Kontext.

    Per object:
      Stage A   : Sketch → LoRA FLUX → obj_img
      Stage MASK: Threshold obj_img → ref_mask
      Stage VLM : VLM(scene, obj_img) → placement description
      Stage COL : paste obj into scene at placement → collage_scene
      Stage K   : FLUX Kontext(reference=collage_scene, prompt) → result

    The collage_scene gives Kontext the full visual prior: WHERE the object
    is, WHAT it looks like, and its approximate SIZE — then FLUX generates
    a version where it is naturally integrated with proper lighting.
    """
    vlm_model, vlm_proc = vlm_pair
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
        print(f"  [A] Generating '{desc}' from sketch ...")
        obj_img = generate_from_sketch(
            pipe=pipe, sketch_path=sketch_path, description=desc,
            seed=seed, num_steps=num_steps, guidance=lora_guidance,
            height=height, width=width, lora_id=lora_id, device=device,
        )
        obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))

        # Stage MASK: extract object mask
        ref_mask = _compute_obj_mask(obj_img)
        n_px = ref_mask.sum()
        print(f"  [MASK] Object pixels: {n_px} ({100*n_px/(height*width):.1f}%)")
        if n_px < 50:
            print("         Fallback: using center 50% crop as object region")
            ref_mask = np.zeros((height, width), dtype=np.uint8)
            ref_mask[height//4:3*height//4, width//4:3*width//4] = 1

        # Stage BBOX: placement mask → bbox (VLM fallback if mask absent)
        mask_path = os.path.join(sketch_dir, f"mask_{name}.png")
        if os.path.isfile(mask_path):
            print(f"  [BBOX] Using placement mask: mask_{name}.png")
            bx1, by1, bx2, by2 = _bbox_from_placement_mask(mask_path, width, height)
            print(f"      bbox: x=[{bx1},{bx2}] y=[{by1},{by2}]")
            with open(os.path.join(out_dir, f"vlm_bbox_{name}.txt"), "w") as f:
                f.write(f"bbox (mask): x1={bx1} y1={by1} x2={bx2} y2={by2}\n")
        else:
            print(f"  [VLM] No placement mask found — running VLM bbox ...")
            bx1, by1, bx2, by2 = _vlm_bbox(
                vlm_model=vlm_model, vlm_processor=vlm_proc,
                scene_img=scene, obj_img=obj_img, description=desc,
                width=width, height=height,
            )
            print(f"      bbox: x=[{bx1},{bx2}] y=[{by1},{by2}]")
            with open(os.path.join(out_dir, f"vlm_bbox_{name}.txt"), "w") as f:
                f.write(f"bbox (vlm): x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

        # Stage COL: build collage scene (AnyDoor's core idea in Kontext)
        print(f"  [COL] Building collage scene (mode={collage_mode}) ...")
        collage_scene, paste_box = build_collage_scene(
            scene        = scene,
            obj_img      = obj_img,
            ref_mask     = ref_mask,
            target_bbox  = (bx1, by1, bx2, by2),
            collage_mode = collage_mode,
            paste_alpha  = paste_alpha,
            feather      = feather,
        )
        collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
        print(f"      Saved collage: collage_{name}.png")

        # Stage K: FLUX Kontext with collage as reference
        blend_p = (
            f"Naturally integrate the {desc} shown in the scene "
            f"with correct lighting, contact shadow, and perspective. "
            f"Do not change any other part of the room."
        )
        print(f"  [K] Kontext integration pass ...")
        print(f"      Prompt: {blend_p[:100]}...")
        with open(os.path.join(out_dir, f"blend_prompt_{name}.txt"), "w") as f:
            f.write(blend_p)

        next_scene = run_standard(
            pipe      = pipe,
            canvas    = collage_scene,   # <── the AnyDoor collage IS the reference
            prompt    = blend_p,
            seed      = seed,
            num_steps = num_steps,
            guidance  = scene_guidance,
            height    = height,
            width     = width,
        )
        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AnyDoor collage method inside FLUX Kontext."
    )
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/phase1_collage")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5,
                   help="Kontext guidance scale for the blending pass. Default 2.5.")
    p.add_argument("--collage_mode",  default="full",
                   choices=["full", "sobel", "blend"],
                   help="What to paste in the collage. "
                        "'full'=actual pixels (default), "
                        "'sobel'=edge map only (AnyDoor-exact), "
                        "'blend'=60%%full+40%%sobel.")
    p.add_argument("--paste_alpha",   type=float, default=0.85,
                   help="Opacity of pasted object in collage (0-1). Default 0.85.")
    p.add_argument("--feather",       type=int,   default=25,
                   help="Gaussian feather radius for soft mask edge (px). Default 25.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--vlm_model",     default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--vlm_device",    default="cpu")
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
    print(f"  phase1_collage_kontext  —  AnyDoor Collage + FLUX Kontext")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Collage mode : {args.collage_mode}  alpha={args.paste_alpha}  "
          f"feather={args.feather}")
    print(f"  Guidance     : {args.guidance}")
    print(f"  VLM          : {args.vlm_model}  [{args.vlm_device}]")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading VLM ...")
    vlm_pair = load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token  = args.hf_token,
        device    = args.device,
        cache_dir = args.cache_dir,
    )

    # Base scene
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    results = run_collage_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        vlm_pair=vlm_pair,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        collage_mode=args.collage_mode,
        paste_alpha=args.paste_alpha,
        feather=args.feather,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
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
