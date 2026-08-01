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
  Stage A   : Sketch → LoRA FLUX → obj_img
  Stage BBOX: Placement mask → bbox           (no VLM — mask_{name}.png required)
  Stage POSE: scene_crop = scene[y1:y2, x1:x2] expanded by 20%
              mini_collage = obj pasted into upscaled crop
              Kontext(mini_collage, pose_prompt) → posed result
              diff(posed, scene_crop) → obj_posed on white bg
  Stage MASK: Threshold obj_posed → ref_mask
  Stage COL : Build collage_scene by pasting obj_posed at placement bbox
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
from typing import Dict, List, Tuple

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


# ── K/V injection: capture + inject obj appearance at target zone ─────────────
#
# Optional second pass on top of the collage:
#   1. 1-step capture: run FLUX with obj_img as canvas, record K/V at TIER_A layers
#   2. Full denoising: run FLUX with collage_scene as canvas,
#      inject captured K/V at target-zone gen tokens
#
# Why this helps on top of collage:
#   The collage gives FLUX the visual prior (shape, colour, position).
#   K/V injection reinforces the appearance signal in attention space at the
#   exact target zone so FLUX doesn't drift the colour/texture during denoising.

_TIER_A        = {0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56}
_SINGLE_TXT_LEN = 512   # T5 sequence length used by FluxKontextPipeline


def _neutralize_white_bg(img: Image.Image, threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    arr[white] = fill
    return Image.fromarray(arr)


def _obj_token_mask(obj_img: Image.Image, h_lat: int, w_lat: int,
                    grey: tuple = (128, 128, 128), tol: int = 10,
                    min_frac: float = 0.05) -> np.ndarray:
    """Bool mask (h_lat*w_lat,): True = object token, False = background."""
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


def _placement_mask_to_token_zone(mask_path: str, height: int, width: int,
                                   pipe) -> np.ndarray:
    """Convert a user-drawn placement mask to a flat bool token mask (n_gen,)."""
    vae_sf    = getattr(pipe, "vae_scale_factor", 8)
    h_lat, w_lat = height // (vae_sf * 2), width // (vae_sf * 2)
    mask_down = Image.open(mask_path).convert("L").resize((w_lat, h_lat), Image.NEAREST)
    zone = (np.array(mask_down) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_lat * w_lat} tokens ({100*zone.mean():.1f}%)")
    return zone


class _KVCapture:
    """
    Attention processor: performs standard attention AND records K/V from the
    Kontext ref-token slice at TIER_A layers during a 1-step capture pass.

    In Kontext hidden_states = [gen_tokens | ref_tokens]. Ref starts at
    img_offset + n_gen.  We store (K, V) on CPU for later injection.
    """
    def __init__(self, n_gen: int):
        self.n_gen = n_gen
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                self.kv[self._layer] = (
                    k[:, :, r_lo:r_hi, :].detach().cpu(),
                    v[:, :, r_lo:r_hi, :].detach().cpu(),
                )

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            self._layer += 1
            return out, enc_out
        self._layer += 1
        return out


class _KVInject:
    """
    Attention processor: injects captured obj K/V into target-zone gen tokens
    at TIER_A layers during the active denoising window (cutoff_frac).
    """
    def __init__(self, n_gen: int,
                 obj_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                 obj_mask_np: np.ndarray,
                 target_zone: np.ndarray, obj_strength: float,
                 n_steps: int, cutoff_frac: Tuple[float, float]):
        self.n_gen       = n_gen
        self.obj_kv      = obj_kv
        self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone   # bool (n_gen,)
        self.obj_strength = obj_strength
        self.n_steps     = n_steps
        self.cutoff_frac = cutoff_frac
        self._layer = 0
        self._step  = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)
        if lo <= self._step < hi and self._layer in self.obj_kv:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off, img_off + self.n_gen

            k_gen = k[:, :, g_lo:g_hi, :].clone()
            v_gen = v[:, :, g_lo:g_hi, :].clone()

            k_obj_raw, v_obj_raw = self.obj_kv[self._layer]
            k_obj_raw = k_obj_raw.to(k.device, k.dtype)
            v_obj_raw = v_obj_raw.to(v.device, v.dtype)

            obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
            if obj_idx.numel() > 0:
                k_obj = k_obj_raw[:, :, obj_idx, :].mean(dim=2, keepdim=True).expand_as(k_gen)
                v_obj = v_obj_raw[:, :, obj_idx, :].mean(dim=2, keepdim=True).expand_as(v_gen)
            else:
                k_obj = k_obj_raw.mean(dim=2, keepdim=True).expand_as(k_gen)
                v_obj = v_obj_raw.mean(dim=2, keepdim=True).expand_as(v_gen)

            tgt   = torch.from_numpy(self.target_zone).to(k.device).view(1, 1, -1, 1)
            s     = self.obj_strength
            k_gen = torch.where(tgt, (1 - s) * k_gen + s * k_obj, k_gen)
            v_gen = torch.where(tgt, (1 - s) * v_gen + s * v_obj, v_gen)

            k = k.clone(); v = v.clone()
            k[:, :, g_lo:g_hi, :] = k_gen
            v[:, :, g_lo:g_hi, :] = v_gen

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            self._layer += 1
            return out, enc_out
        self._layer += 1
        return out


@torch.no_grad()
def run_with_kv_injection(
    pipe,
    canvas:       Image.Image,
    prompt:       str,
    obj_img:      Image.Image,
    obj_mask_np:  np.ndarray,
    target_zone:  np.ndarray,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    height:       int,
    width:        int,
    obj_strength: float = 0.7,
    cutoff_frac:  Tuple[float, float] = (0.0, 0.6),
) -> Image.Image:
    """
    FLUX Kontext with K/V injection from obj_img at target_zone tokens.

    Step 1 — 1-step capture pass:
        obj_img (white bg neutralised) as canvas → record K/V at TIER_A layers
        from the ref-token slice.

    Step 2 — Full denoising (num_steps):
        canvas (the collage scene) as reference → at TIER_A layers, blend
        captured K/V into gen tokens that fall inside target_zone.
        Injection active for the first cutoff_frac fraction of steps.

    The captured K/V encode the object's appearance at the feature level.
    The collage already placed the object visually — this reinforces that
    appearance signal in attention space so FLUX doesn't drift it during denoising.
    """
    vae_sf      = getattr(pipe, "vae_scale_factor", 8)
    h_lat       = height // (vae_sf * 2)
    w_lat       = width  // (vae_sf * 2)
    n_gen       = h_lat * w_lat

    orig_procs  = pipe.transformer.attn_processors   # save to restore later

    # ── Stage K-cap: capture K/V from obj_img ────────────────────────────────
    obj_neutral  = _neutralize_white_bg(obj_img)
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(
        image=obj_neutral,
        prompt="photorealistic object, studio lighting",
        num_inference_steps=1, guidance_scale=1.0,
        height=height, width=width, max_sequence_length=512,
        generator=gen0, output_type="latent",
    )
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")

    # ── Stage K-inj: denoising with obj K/V injection ────────────────────────
    inject_proc = _KVInject(
        n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
        target_zone=target_zone, obj_strength=obj_strength,
        n_steps=num_steps, cutoff_frac=cutoff_frac,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_callback(pipe_ref, step_index, timestep, callback_kwargs):
        inject_proc._step  = step_index + 1
        inject_proc._layer = 0
        return callback_kwargs

    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=canvas, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, max_sequence_length=512,
        generator=gen1, output_type="pil",
        callback_on_step_end=_step_callback,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]

    pipe.transformer.set_attn_processor(orig_procs)   # restore
    return result


# ── Stage POSE: Kontext-based pose adjustment ────────────────────────────────

@torch.no_grad()
def _derive_pose_string(
    bbox:        Tuple[int, int, int, int],
    scene_w:     int,
    scene_h:     int,
    description: str,
) -> str:
    """
    Convert placement mask bbox geometry into a natural-language pose instruction.

    Horizontal position → which direction the object faces (objects face inward).
    Vertical position   → camera elevation (high in frame = more top-down).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / scene_w   # 0=left, 1=right
    cy = (y1 + y2) / 2 / scene_h   # 0=top,  1=bottom

    if cx < 0.33:
        facing = "facing toward the right at a 3/4 angle"
    elif cx > 0.67:
        facing = "facing toward the left at a 3/4 angle"
    else:
        facing = "facing directly toward the viewer at a slight 3/4 angle"

    if cy < 0.40:
        elevation = "seen from above with a top-down perspective"
    elif cy > 0.70:
        elevation = "at eye level with a straight-on perspective"
    else:
        elevation = "seen from a slight overhead angle"

    return (
        f"Rotate the {description} so it is {facing}, {elevation}. "
        f"Keep the {description} perfectly identical — same color, same design, "
        f"same details. Only change the viewing angle. "
        f"White background, studio lighting, no shadows."
    )


def _pose_object_in_context(
    pipe,
    obj_img:     Image.Image,
    bbox:        Tuple[int, int, int, int],
    description: str,
    seed:        int,
    num_steps:   int,
    guidance:    float,
    height:      int,
    width:       int,
) -> Image.Image:
    """
    Repose obj_img using Kontext with object-only input.

    Passes obj_img directly as the Kontext canvas (no scene crop, no collage).
    Kontext edits the viewing angle while preserving object identity.
    Pose string is derived from bbox geometry — no VLM needed.

    Falls back to original obj_img if the output is indistinguishable from input.
    """
    pose_prompt = _derive_pose_string(bbox, width, height, description)
    print(f"    [POSE] Pose prompt: {pose_prompt[:120]}")

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    posed_pil = pipe(
        image=obj_img,
        prompt=pose_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=height, width=width,
        max_sequence_length=512,
        generator=gen,
        output_type="pil",
    ).images[0]

    # Sanity check: if output is too similar to input, return original
    diff = np.abs(
        np.array(posed_pil.convert("RGB"), dtype=np.int32) -
        np.array(obj_img.convert("RGB"),   dtype=np.int32)
    ).mean()
    print(f"    [POSE] Input/output mean diff: {diff:.1f}")
    if diff < 2.0:
        print("    [POSE] Output nearly identical to input — returning original obj_img")
        return obj_img

    return posed_pil


# ── Sobel edge extraction (mirrors AnyDoor's sobel()) ────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    """
    AnyDoor-style high-frequency detail map (datasets/data_utils.py::sobel).
    Sobel-filtered RGB: texture edges where object is, black elsewhere.
    thresh=30 (vs AnyDoor's 50) because LoRA-generated objects have smoother
    gradients than real photos — 50 produces all-black on synthetic output.
    Erosion guard: falls back to un-eroded mask if erosion kills the object.
    """
    H, W = img.shape[:2]
    small   = cv2.resize(img,  (256, 256))
    mask_s  = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    kernel  = np.ones((5, 5), np.uint8)
    eroded  = cv2.erode(mask_s, kernel, iterations=2)
    mask_s  = eroded if eroded.any() else mask_s

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
      'full'  — paste actual obj pixels, feathered alpha blend (colour + texture)
      'sobel' — AnyDoor exact: Sobel detail map hard-pasted at full bbox, no
                alpha blend; Kontext handles blending (thresh=50, as in paper)
      'blend' — 60% full + 40% sobel, feathered alpha blend
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

    # AnyDoor-exact sobel: resize obj to fill full target zone, hard paste,
    # no alpha blending — Kontext (our "ControlNet") handles the blending.
    if collage_mode == "sobel":
        obj_z      = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z     = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        print(f"    [COLLAGE] AnyDoor hard paste {zone_w}×{zone_h} Sobel at "
              f"scene[{y1}:{y2},{x1}:{x2}]")
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    # Scale obj to fill ~80% of the target zone (preserve aspect ratio)
    oh, ow  = obj_crop.shape[:2]
    scale   = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h   = max(4, int(oh * scale))
    new_w   = max(4, int(ow * scale))

    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h))
    mask_rs = (mask_rs > 0.5).astype(np.float32)

    # Choose what to paste
    if collage_mode == "blend":
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


# ── Stage HAR: image harmonization ───────────────────────────────────────────
#
# Two paths:
#   PCT-Net  — full harmonization network (CVPR 2023, rakutentech/PCT-Net-Image-Harmonization)
#              Requires: git clone the repo, pass --pctnet_dir and --pctnet_weights.
#   Reinhard — luminance-only Lab color transfer, no external deps, instant fallback.
#
# Harmonization mask is derived by diffing collage_scene vs original scene so
# it works correctly for all collage modes (full / sobel / blend).


def _collage_obj_mask(collage: Image.Image, scene: Image.Image,
                       threshold: int = 10) -> np.ndarray:
    """
    Binary uint8 mask (H, W): 1 where collage differs from scene (= pasted obj).
    More reliable than forwarding ref_mask because the paste location shifts
    (80% scale + centering) relative to the original obj_img space.
    """
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask


def _harmonize_reinhard(composite: Image.Image, obj_mask: np.ndarray) -> Image.Image:
    """
    Luminance-only Reinhard Lab transfer.
    Matches the pasted object's lightness statistics to the surrounding background
    while keeping the object's original color (A/B channels untouched).
    Soft-blends at the mask boundary to avoid hard edges.
    """
    comp_np  = np.array(composite.convert("RGB"))
    comp_lab = cv2.cvtColor(comp_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    bg_L = comp_lab[obj_mask == 0, 0]
    fg_L = comp_lab[obj_mask >  0, 0]

    if len(bg_L) < 100 or len(fg_L) < 100:
        print("    [HAR] Reinhard: insufficient pixels — skipping")
        return composite

    fg_norm = (fg_L - fg_L.mean()) / (fg_L.std() + 1e-6)
    comp_lab[obj_mask > 0, 0] = np.clip(
        fg_norm * bg_L.std() + bg_L.mean(), 0, 255
    )

    result_np = cv2.cvtColor(comp_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Feather boundary
    mask_f = cv2.GaussianBlur(obj_mask.astype(np.float32), (31, 31), 10)
    blended = (result_np.astype(np.float32) * mask_f[:, :, None] +
               comp_np.astype(np.float32)   * (1 - mask_f[:, :, None]))
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _harmonize_pctnet(
    composite: Image.Image,
    obj_mask:  np.ndarray,
    pctnet_dir: str,
    weights_path: str,
    device: str = "cuda",
) -> Image.Image:
    """
    PCT-Net harmonization (CVPR 2023).

    Setup:
      git clone https://github.com/rakutentech/PCT-Net-Image-Harmonization
      pip install -r PCT-Net-Image-Harmonization/requirements.txt
    Then pass:
      --pctnet_dir  /path/to/PCT-Net-Image-Harmonization
      --pctnet_weights /path/to/PCT-Net-Image-Harmonization/pretrained_models/PCTNet_ViT.pth
    """
    import sys as _sys
    if pctnet_dir not in _sys.path:
        _sys.path.insert(0, pctnet_dir)

    from iharm.inference.utils import load_model
    from iharm.inference.predictor import Predictor
    from iharm.mconfigs import ALL_MCONFIGS

    model_type = "ViT_pct"
    net = load_model(model_type, weights_path, verbose=False)
    cfg = ALL_MCONFIGS[model_type]
    norm = cfg.get("params", {}).get("input_normalization",
                                     {"mean": (.485, .456, .406),
                                      "std":  (.229, .224, .225)})
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    predictor = Predictor(net, dev, mean=norm["mean"], std=norm["std"],
                           with_flip=False)

    comp_np  = np.array(composite.convert("RGB"))
    mask_f32 = obj_mask.astype(np.float32)   # [0, 1]

    comp_lr  = cv2.resize(comp_np,  (256, 256))
    mask_lr  = cv2.resize(mask_f32, (256, 256))

    _, result_np = predictor.predict(comp_lr, comp_np, mask_lr, mask_f32)
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    return Image.fromarray(result_np)


def _harmonize(
    composite:    Image.Image,
    obj_mask:     np.ndarray,
    pctnet_dir:   str = "",
    pctnet_weights: str = "",
    device:       str = "cuda",
) -> Image.Image:
    """
    Harmonize the pasted object with the background.
    Uses PCT-Net when --pctnet_dir and --pctnet_weights are provided,
    otherwise falls back to Reinhard Lab luminance transfer.
    """
    if pctnet_dir and pctnet_weights and \
            os.path.isdir(pctnet_dir) and os.path.isfile(pctnet_weights):
        try:
            print("    [HAR] PCT-Net harmonization ...")
            result = _harmonize_pctnet(composite, obj_mask, pctnet_dir,
                                        pctnet_weights, device)
            print("    [HAR] PCT-Net done")
            return result
        except Exception as e:
            print(f"    [HAR] PCT-Net failed ({e!s:.120}) — falling back to Reinhard")

    print("    [HAR] Reinhard Lab luminance transfer ...")
    return _harmonize_reinhard(composite, obj_mask)


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
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
    use_kv:         bool  = False,
    obj_strength:   float = 0.7,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.6),
    pose_steps:     int   = 15,
    pose_guidance:  float = 2.5,
    skip_pose:      bool  = False,
    enh_steps:      int   = 6,
    enh_guidance:   float = 1.5,
    skip_enh:       bool  = False,
    pctnet_dir:     str   = "",
    pctnet_weights: str   = "",
) -> List[Image.Image]:
    """
    Incremental object insertion via FLUX Kontext with Kontext-based pose adjustment.

    Per object:
      Stage A   : Sketch → LoRA FLUX → obj_img
      Stage BBOX: Placement mask → bbox  (mask_{name}.png required)
      Stage POSE: Kontext(scene_crop+obj, pose_prompt) → obj_posed
                  diff extraction → obj on white bg with natural pose
      Stage MASK: Threshold obj_posed → ref_mask
      Stage COL : paste obj_posed into scene at bbox → collage_scene
      Stage HAR : Harmonize pasted object with background (PCT-Net or Reinhard)
      Stage K   : FLUX Kontext(reference=collage_scene, prompt) → result
    """
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

        # Stage POSE: Kontext repose object-only, angle derived from mask geometry
        if not skip_pose:
            print(f"  [POSE] Adjusting pose via Kontext (object-only) ...")
            obj_posed = _pose_object_in_context(
                pipe=pipe, obj_img=obj_img,
                bbox=(bx1, by1, bx2, by2), description=desc,
                seed=seed, num_steps=pose_steps, guidance=pose_guidance,
                height=height, width=width,
            )
            obj_posed.save(os.path.join(out_dir, f"obj_posed_{name}.png"))
            print(f"      Saved: obj_posed_{name}.png")
        else:
            obj_posed = obj_img

        # Stage MASK: extract mask from posed object
        ref_mask = _compute_obj_mask(obj_posed)
        n_px = ref_mask.sum()
        print(f"  [MASK] Object pixels: {n_px} ({100*n_px/(height*width):.1f}%)")
        if n_px < 50:
            print("         Fallback: using center 50% crop as object region")
            ref_mask = np.zeros((height, width), dtype=np.uint8)
            ref_mask[height//4:3*height//4, width//4:3*width//4] = 1

        # Stage COL: build collage scene with posed object
        print(f"  [COL] Building collage scene (mode={collage_mode}) ...")
        collage_scene, paste_box = build_collage_scene(
            scene        = scene,
            obj_img      = obj_posed,
            ref_mask     = ref_mask,
            target_bbox  = (bx1, by1, bx2, by2),
            collage_mode = collage_mode,
            paste_alpha  = paste_alpha,
            feather      = feather,
        )
        collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
        print(f"      Saved collage: collage_{name}.png")

        # Stage HAR: harmonize pasted object with background
        print(f"  [HAR] Harmonizing object with scene ...")
        obj_mask_har = _collage_obj_mask(collage_scene, scene)
        n_har = int(obj_mask_har.sum())
        print(f"      HAR mask: {n_har} px ({100*n_har/obj_mask_har.size:.1f}%)")
        if n_har >= 50:
            collage_scene = _harmonize(
                composite=collage_scene,
                obj_mask=obj_mask_har,
                pctnet_dir=pctnet_dir,
                pctnet_weights=pctnet_weights,
                device=device,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_har_{name}.png"))
            print(f"      Saved: collage_har_{name}.png")
        else:
            print("      HAR mask too sparse — skipping harmonization")

        # Stage K: FLUX Kontext integration pass
        blend_p = (
            f"Naturally integrate the {desc} shown in the scene "
            f"with correct lighting, contact shadow, and perspective. "
            f"Do not change any other part of the room."
        )
        print(f"  [K] Kontext integration pass (kv={use_kv}) ...")
        print(f"      Prompt: {blend_p[:100]}...")
        with open(os.path.join(out_dir, f"blend_prompt_{name}.txt"), "w") as f:
            f.write(blend_p)

        if use_kv:
            vae_sf   = getattr(pipe, "vae_scale_factor", 8)
            h_lat    = height // (vae_sf * 2)
            w_lat    = width  // (vae_sf * 2)
            tok_mask = _obj_token_mask(obj_posed, h_lat, w_lat)
            pct      = 100.0 * tok_mask.mean()
            print(f"      obj token mask: {tok_mask.sum()} / {h_lat*w_lat} ({pct:.1f}%)")
            target_zone = _placement_mask_to_token_zone(mask_path, height, width, pipe)
            next_scene = run_with_kv_injection(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                obj_img=obj_posed, obj_mask_np=tok_mask, target_zone=target_zone,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
                obj_strength=obj_strength, cutoff_frac=cutoff_frac,
            )
        else:
            next_scene = run_standard(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
            )

        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        # Stage ENH: second Kontext pass to remove blending artifacts
        if not skip_enh:
            print(f"  [ENH] Refinement pass ({enh_steps} steps, guidance={enh_guidance}) ...")
            enh_prompt = (
                "Photorealistic interior room. Restore sharp floor texture, "
                "remove blending artifacts, preserve all objects and colors exactly."
            )
            next_scene = run_standard(
                pipe=pipe, canvas=next_scene, prompt=enh_prompt,
                seed=seed, num_steps=enh_steps, guidance=enh_guidance,
                height=height, width=width,
            )
            enh_path = os.path.join(out_dir, f"result_enh_{name}.png")
            next_scene.save(enh_path)
            print(f"      Saved: {enh_path}")

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
    p.add_argument("--feather",       type=int,   default=10,
                   help="Gaussian feather radius for soft mask edge (px). Default 10.")
    p.add_argument("--kv_injection",  action="store_true",
                   help="Enable K/V injection from obj_img at target zone on top of collage.")
    p.add_argument("--obj_strength",  type=float, default=0.7,
                   help="K/V injection strength at target zone (0-1). Default 0.7.")
    p.add_argument("--cutoff_frac",   default="0.0,0.6",
                   help="Injection active window as 'start,end' step fractions. Default '0.0,0.6'.")
    p.add_argument("--skip_pose",      action="store_true",
                   help="Skip Stage POSE; use raw obj_img directly in collage.")
    p.add_argument("--pose_steps",    type=int,   default=15,
                   help="Kontext steps for the pose-adjustment pass. Default 15.")
    p.add_argument("--pose_guidance", type=float, default=2.5,
                   help="Guidance scale for the pose pass. Default 2.5.")
    p.add_argument("--skip_enh",       action="store_true",
                   help="Skip Stage ENH refinement pass.")
    p.add_argument("--enh_steps",     type=int,   default=6,
                   help="Kontext steps for the refinement pass. Default 6.")
    p.add_argument("--enh_guidance",  type=float, default=1.5,
                   help="Guidance scale for the refinement pass. Default 1.5.")
    p.add_argument("--pctnet_dir",    default="",
                   help="Path to cloned rakutentech/PCT-Net-Image-Harmonization repo. "
                        "If empty, Reinhard Lab transfer is used instead.")
    p.add_argument("--pctnet_weights", default="",
                   help="Path to PCTNet_ViT.pth weights file. "
                        "If empty, Reinhard Lab transfer is used instead.")
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
    print(f"  phase1_collage_kontext  —  AnyDoor Collage + FLUX Kontext")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Collage mode : {args.collage_mode}  alpha={args.paste_alpha}  feather={args.feather}")
    print(f"  Guidance     : {args.guidance}")
    print(f"  Pose pass    : {'skip' if args.skip_pose else f'steps={args.pose_steps}  guidance={args.pose_guidance}'}")
    har_backend = "PCT-Net" if (args.pctnet_dir and args.pctnet_weights) else "Reinhard (fallback)"
    print(f"  Harmonization: {har_backend}")
    print(f"  Refinement   : {'skip' if args.skip_enh else f'steps={args.enh_steps}  guidance={args.enh_guidance}'}")
    print(f"  KV injection : {args.kv_injection}"
          + (f"  strength={args.obj_strength}  cutoff={args.cutoff_frac}"
             if args.kv_injection else ""))
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading FLUX.1-Kontext-dev ...")
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

    _cf = [float(x) for x in args.cutoff_frac.split(",")]
    cutoff_frac: Tuple[float, float] = (_cf[0], _cf[1])

    results = run_collage_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        collage_mode=args.collage_mode,
        paste_alpha=args.paste_alpha,
        feather=args.feather,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_kv=args.kv_injection,
        obj_strength=args.obj_strength,
        cutoff_frac=cutoff_frac,
        pose_steps=args.pose_steps,
        pose_guidance=args.pose_guidance,
        skip_pose=args.skip_pose,
        enh_steps=args.enh_steps,
        enh_guidance=args.enh_guidance,
        skip_enh=args.skip_enh,
        pctnet_dir=args.pctnet_dir,
        pctnet_weights=args.pctnet_weights,
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
