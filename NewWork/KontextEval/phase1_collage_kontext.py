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
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ── sibling imports ───────────────────────────────────────────────────────────
_base = Path(__file__).parent

def _load_mod(name: str):
    p   = _base / f"{name}.py"
    sp  = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod

_comp = _load_mod("phase1_composite")
_vlm  = _load_mod("phase1_sketch_vlm")
_sk   = _load_mod("phase1_sketch")

run_standard         = _comp.run_standard
save_grid            = _comp.save_grid
generate_from_sketch = _vlm.generate_from_sketch
load_vlm             = _sk.load_vlm

sys.path.insert(0, str(_base.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "white ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT = "A modern living room with a sofa and a wooden coffee table."
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


# ── Combined VLM call: prompt + placement bbox in one pass ───────────────────

def _vlm_prompt_and_bbox(
    vlm_model,
    vlm_processor,
    scene_img:   Image.Image,
    obj_img:     Image.Image,
    description: str,
    width:       int,
    height:      int,
    max_new_tokens: int = 512,
) -> tuple[str, Tuple[int, int, int, int]]:
    """
    Single VLM call that returns BOTH the Kontext blending prompt AND the
    placement bounding box from one shared reasoning pass.

    Why one call instead of two:
    - Two separate calls are independent inferences that can contradict each
      other: the text might say "left wall" while the bbox lands center-right.
    - One call shares the same reasoning: wherever the model decides to place
      the object in text, it grounds the bbox to that exact same location.

    Output format the model is asked to produce:
        PROMPT: <text describing appearance, placement, orientation, scale>
        BBOX: <|box_start|>(x1,y1),(x2,y2)<|box_end|>

    Bbox coords are normalised [0,1000) → rescaled to pixel space.
    Parsing is tolerant: bbox tokens are extracted independently of text
    structure, so minor format deviations still yield valid coordinates.

    Returns (prompt_text, (x1, y1, x2, y2)).
    Fallback bbox: center-floor zone if no grounding tokens found.
    Fallback prompt: raw model output if "PROMPT:" prefix missing.
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

    instruction = (
        f"You are an expert image composition assistant.\n\n"
        f"Image 1: A room scene where I want to insert a new object.\n"
        f"Image 2: The {description} I want to insert (on a white background).\n\n"
        f"Decide WHERE in the room to place the {description}, then output "
        f"EXACTLY two lines:\n\n"
        f"PROMPT: <a single complete sentence describing: the {description}'s "
        f"exact appearance from Image 2, the chosen placement location in the "
        f"room, orientation and surface contact, realistic scale, ending with: "
        f"'Do not change any other part of the room.'>\n"
        f"BBOX: <bounding box of that exact placement location in Image 1>\n\n"
        f"The PROMPT and BBOX must describe the SAME location. "
        f"Output nothing else."
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
    images_in_order = [scene_small, obj_small]
    inputs = vlm_processor(
        text=[text],
        images=images_in_order,
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
    # skip_special_tokens=False so <|box_start|>/<|box_end|> survive
    raw = vlm_processor.batch_decode(
        out_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print(f"    [VLM] raw output:\n      {raw!r}")

    # ── Extract prompt text ───────────────────────────────────────────────────
    pm = re.search(r'PROMPT:\s*(.*?)(?=\nBBOX:|\Z)', raw, re.DOTALL | re.IGNORECASE)
    if pm:
        prompt_text = pm.group(1).strip()
    else:
        # Tolerant fallback: strip bbox line if present, use rest as prompt
        prompt_text = re.sub(
            r'\nBBOX:.*', '', raw, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        if not prompt_text:
            prompt_text = (
                f"Naturally integrate the {description} into the room scene. "
                f"Do not change any other part of the room."
            )

    # Strip stray special tokens from prompt text
    prompt_text = re.sub(r'<\|[^|]+\|>', '', prompt_text).strip()

    # Resized scene dimensions — needed to rescale plain-text coords back
    sw, sh = scene_small.size

    def _rescale(rx1, ry1, rx2, ry2):
        x1 = max(0,      int(rx1 * width  / sw))
        y1 = max(0,      int(ry1 * height / sh))
        x2 = min(width,  int(rx2 * width  / sw))
        y2 = min(height, int(ry2 * height / sh))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    # ── Parser 1: native Qwen grounding tokens <|box_start|>(x,y),(x,y)<|box_end|>
    # Coords normalised [0, 1000) relative to the resized image.
    bm = re.search(
        r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>',
        raw,
    )
    if bm:
        x1n, y1n, x2n, y2n = (int(bm.group(i)) for i in range(1, 5))
        bbox = _rescale(
            int(x1n * sw / 1000), int(y1n * sh / 1000),
            int(x2n * sw / 1000), int(y2n * sh / 1000),
        )
        if bbox:
            print(f"    [VLM] bbox (grounding tokens): x=[{bbox[0]},{bbox[2]}] y=[{bbox[1]},{bbox[3]}]")
            return prompt_text, bbox

    # ── Parser 2: plain [x1, y1, x2, y2] — model outputs pixel coords in
    # resized image space when asked to produce a BBOX: line as text.
    bm2 = re.search(
        r'BBOX\s*:\s*\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]',
        raw, re.IGNORECASE,
    )
    if bm2:
        rx1, ry1, rx2, ry2 = (int(bm2.group(i)) for i in range(1, 5))
        bbox = _rescale(rx1, ry1, rx2, ry2)
        if bbox:
            print(f"    [VLM] bbox (plain): x=[{bbox[0]},{bbox[2]}] y=[{bbox[1]},{bbox[3]}]")
            return prompt_text, bbox

    print(f"    [VLM] bbox parse failed — center-floor fallback")
    return prompt_text, (width // 6, height // 2, 5 * width // 6, height)


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


# ── Kontext blending prompt ───────────────────────────────────────────────────

def _blend_prompt(obj_description: str, vlm_placement: str) -> str:
    """
    Build a Kontext blending prompt that stays under CLIP's 77-token limit.

    Takes the first sentence of the VLM output and caps it at 35 words so
    the full prompt stays ~60 tokens (first sentence ~35 words + fixed
    integration suffix ~25 tokens = safely under 77).
    """
    first_sentence = vlm_placement.split(".")[0].strip()
    words = first_sentence.split()
    if len(words) > 35:
        first_sentence = " ".join(words[:35])
    return (
        f"{first_sentence}. "
        f"Naturally integrate the {obj_description} — "
        f"blend with correct perspective, shadows, and lighting. "
        f"Do not change any other part of the room."
    )


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

        # Stage VLM: prompt + bbox in one consistent pass
        print(f"  [VLM] Generating prompt and placement bbox ...")
        vlm_prompt, (bx1, by1, bx2, by2) = _vlm_prompt_and_bbox(
            vlm_model=vlm_model, vlm_processor=vlm_proc,
            scene_img=scene, obj_img=obj_img, description=desc,
            width=width, height=height,
        )
        print(f"\n  {_SEP}")
        print(f"  [VLM → Prompt]  {name}")
        print(f"  {_SEP}")
        for line in vlm_prompt.splitlines():
            print(f"  {line}")
        print(f"  {_SEP}\n")
        with open(os.path.join(out_dir, f"vlm_prompt_{name}.txt"), "w") as f:
            f.write(f"{vlm_prompt}\n\nbbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

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
        blend_p = _blend_prompt(desc, vlm_prompt)
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
