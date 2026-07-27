# -*- coding: utf-8 -*-
"""
Phase 1 -- Sketch-Conditioned Multi-Step Object Insertion

Pipeline
--------
Two-stage, training-free:

  Stage A  Sketch -> Object Image
    Given a hand-drawn sketch of an object, generate a photorealistic version
    of it on a plain white background.

    Mode 'lora'  (default):
      Uses FluxKontextPipeline + gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA.
      Single model throughout (no swap), trigger phrase forces structure fidelity.

    Mode 'controlnet':
      Uses FluxControlNetPipeline (FLUX.1-dev + lineart ControlNet).
      Stronger geometric control; requires loading a second large model and
      swapping it out before Stage B.

  Stage B  Object -> Scene  (shared by both modes)
    1. PROBE  -- find where in the accumulated scene the prompt would place the object
    2. PLACE  -- paste the Stage-A object at the probe-detected location (rough composite)
    3. REFINE -- run Kontext K/V injection with rough_composite as canvas
                  (Kontext smooths seams and adjusts lighting; K/V injection locks background)
    4. BG-REPLACE -- overwrite pure-background pixels with the original base (zero drift)

Mask extraction
---------------
RMBG-2.0 (briaai/RMBG-2.0, BiRefNet) replaces naive pixel-diff-vs-white.
Works regardless of object colour (handles white vase on white background).

Usage
-----
  # LoRA mode (single model, simplest)
  python NewWork/KontextEval/phase1_sketch.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --out_dir results/phase1_sketch --mode lora

  # ControlNet mode (stronger shape control)
  python NewWork/KontextEval/phase1_sketch.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --out_dir results/phase1_sketch --mode controlnet

  # Both modes + text-only bg_replace for comparison
  python NewWork/KontextEval/phase1_sketch.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --out_dir results/phase1_sketch --mode all

Sketch format
-------------
  inputs/sketches/bicycle.png   -- dark lines on light background (standard paper sketch)
  inputs/sketches/vase.png
  inputs/sketches/ball.png
  (name must match the 'name' field in EDITS or --config)
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── import utilities from phase1_composite (avoid duplication) ────────────────
_comp_path = str(Path(__file__).parent / "phase1_composite.py")
_spec = importlib.util.spec_from_file_location("phase1_composite", _comp_path)
_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comp)

pixel_to_token_mask  = _comp.pixel_to_token_mask
_token_to_pixel      = _comp._token_to_pixel
color_overlay        = _comp.color_overlay
region_diff          = _comp.region_diff
run_standard         = _comp.run_standard
save_grid            = _comp.save_grid
feather_mask         = _comp.feather_mask          # has interior=1.0 clamp fix
run_kv_injected      = _comp.run_kv_injected
build_stability_grid = _comp.build_stability_grid
run_bg_replace_chain = _comp.run_bg_replace_chain

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline

_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)
from kontext_injection import TIER_A, TIER_ALL

# ============================================================
# Edit list
# ============================================================

EDITS: List[dict] = [
    {
        "name": "bicycle",
        "prompt": (
            "Add a yellow bicycle leaning against the wall on the left side. "
            "Keep the rest of the room exactly the same."
        ),
        "description": "yellow bicycle",
    },
    {
        "name": "vase",
        "prompt": (
            "Add a white ceramic vase with flowers on the coffee table. "
            "Keep the rest of the room exactly the same."
        ),
        "description": "white ceramic vase with flowers",
    },
    {
        "name": "ball",
        "prompt": (
            "Add a yellow round ball in the right corner of the room next to the sofa. "
            "Keep the rest of the room exactly the same."
        ),
        "description": "yellow round ball",
    },
]

BASE_PROMPT   = "A modern living room with a sofa and a wooden coffee table."
TIER_A_LAYERS = list(TIER_A)
ALL_LAYERS    = list(TIER_ALL)

LORA_TRIGGER  = "Convert this sketch into real life version, follow exact structure."

# ============================================================
# RMBG-2.0 (BiRefNet) -- semantic background removal
# ============================================================

def load_rmbg(cache_dir: str, device: str = "cuda"):
    """Load briaai/RMBG-2.0 for semantic foreground mask extraction."""
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to(device).eval()
    return model


def rmbg_extract_mask(model, obj_img: Image.Image,
                      threshold: float = 0.5,
                      device: str = "cuda") -> np.ndarray:
    """
    Semantic foreground mask via RMBG-2.0.
    Returns bool array (H, W) -- True where the object is.
    Works on white objects on white backgrounds (unlike pixel-diff).
    """
    from torchvision import transforms as T

    tf = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = tf(obj_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)[-1].sigmoid().cpu().squeeze()
    mask_pil = T.ToPILImage()(pred).resize(obj_img.size, Image.BILINEAR)
    return np.array(mask_pil) > int(threshold * 255)


# ============================================================
# Sketch preprocessing (ControlNet mode only)
# ============================================================

def preprocess_sketch(sketch_path: str, size: int = 1024) -> Image.Image:
    """
    Prepare a hand-drawn sketch for lineart ControlNet input.
    Converts black-on-white to white-on-black (ControlNet expected format).
    """
    img = Image.open(sketch_path).convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(img)
    if arr.mean() > 128:          # dark lines on light background -> invert
        arr = 255 - arr
    return Image.fromarray(arr).convert("RGB")


# ============================================================
# Stage A helpers
# ============================================================

def stage_a_lora(
    pipe,
    edits: List[dict],
    sketch_dir: str,
    lora_id: str,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    rmbg_model,
    rmbg_threshold: float,
    out_dir: str,
    device: str,
) -> Dict[str, Tuple[Image.Image, np.ndarray]]:
    """
    LoRA mode Stage A: pass sketch as Kontext reference + LoRA trigger phrase.
    Returns {name: (obj_img, obj_mask)} for each edit.
    """
    print("\n=== Stage A: Sketch -> Object (LoRA mode) ===")
    print(f"  Loading LoRA: {lora_id}")
    pipe.load_lora_weights(lora_id)

    results: Dict[str, Tuple[Image.Image, np.ndarray]] = {}
    for edit in edits:
        name   = edit["name"]
        desc   = edit.get("description", name)
        sketch_path = os.path.join(sketch_dir, f"{name}.png")

        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}\n"
                f"Create a hand-drawn sketch and save it as '{name}.png' in --sketch_dir."
            )

        sketch_pil = Image.open(sketch_path).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )
        prompt_a = (
            f"{LORA_TRIGGER} {desc} on a plain white background, "
            "no shadows, studio lighting, photorealistic."
        )
        print(f"  [{name}] generating from sketch ({num_steps} steps, g={guidance:.1f}) ...")
        generator = torch.Generator(device=device).manual_seed(seed)
        obj_img = pipe(
            image=sketch_pil,
            prompt=prompt_a,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height, width=width,
            max_sequence_length=512,
            generator=generator,
            output_type="pil",
        ).images[0]
        obj_img.save(os.path.join(out_dir, f"sketch_gen_{name}.png"))

        print(f"  [{name}] extracting mask (RMBG-2.0) ...")
        obj_mask = rmbg_extract_mask(rmbg_model, obj_img, rmbg_threshold, device)
        _save_mask_preview(obj_img, obj_mask,
                           os.path.join(out_dir, f"sketch_mask_{name}.png"))
        results[name] = (obj_img, obj_mask)

    print("  Unloading LoRA ...")
    pipe.unload_lora_weights()
    return results


def stage_a_controlnet(
    edits: List[dict],
    sketch_dir: str,
    controlnet_id: str,
    controlnet_scale: float,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    rmbg_model,
    rmbg_threshold: float,
    out_dir: str,
    hf_token: str,
    cache_dir: str,
    device: str,
) -> Dict[str, Tuple[Image.Image, np.ndarray]]:
    """
    ControlNet mode Stage A: FLUX.1-dev + lineart ControlNet.
    Loads the ControlNet pipeline, generates all objects, then unloads
    before returning so GPU memory is free for the Kontext pipeline.
    """
    print("\n=== Stage A: Sketch -> Object (ControlNet mode) ===")

    # Diffusers version check (FluxControlNetPipeline requires >= 0.31)
    import diffusers as _diff
    ver = tuple(int(x) for x in _diff.__version__.split(".")[:2])
    if ver < (0, 31):
        raise RuntimeError(
            f"FluxControlNetPipeline requires diffusers>=0.31, got {_diff.__version__}.\n"
            "Upgrade with: pip install -U diffusers"
        )

    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    print(f"  Loading ControlNet: {controlnet_id}")
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        token=hf_token,
    )
    pipe_cn = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        token=hf_token,
    ).to(device)

    results: Dict[str, Tuple[Image.Image, np.ndarray]] = {}
    for edit in edits:
        name   = edit["name"]
        desc   = edit.get("description", name)
        sketch_path = os.path.join(sketch_dir, f"{name}.png")

        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}\n"
                f"Create a hand-drawn sketch and save it as '{name}.png' in --sketch_dir."
            )

        edge_map = preprocess_sketch(sketch_path, size=width)
        edge_map.save(os.path.join(out_dir, f"sketch_edge_{name}.png"))

        prompt_a = (
            f"{desc} on a plain white background, "
            "no shadows, studio lighting, photorealistic, high quality."
        )
        print(f"  [{name}] generating from sketch ({num_steps} steps, "
              f"cn_scale={controlnet_scale}) ...")
        generator = torch.Generator(device=device).manual_seed(seed)
        obj_img = pipe_cn(
            prompt=prompt_a,
            control_image=edge_map,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height, width=width,
            generator=generator,
            output_type="pil",
        ).images[0]
        obj_img.save(os.path.join(out_dir, f"sketch_gen_{name}.png"))

        print(f"  [{name}] extracting mask (RMBG-2.0) ...")
        obj_mask = rmbg_extract_mask(rmbg_model, obj_img, rmbg_threshold, device)
        _save_mask_preview(obj_img, obj_mask,
                           os.path.join(out_dir, f"sketch_mask_{name}.png"))
        results[name] = (obj_img, obj_mask)

    print("  Unloading ControlNet pipeline ...")
    del pipe_cn, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _save_mask_preview(obj_img: Image.Image, mask: np.ndarray, path: str):
    """Save a side-by-side preview: generated object + extracted mask."""
    W, H = obj_img.size
    mask_preview = Image.fromarray(mask.astype(np.uint8) * 255, "L").convert("RGB")
    combined = Image.new("RGB", (W * 2, H))
    combined.paste(obj_img, (0, 0))
    combined.paste(mask_preview, (W, 0))
    combined.save(path)


# ============================================================
# Placement: generated object -> scene bounding box
# ============================================================

def paste_object_at_probe(
    scene: Image.Image,
    obj_img: Image.Image,
    obj_mask: np.ndarray,          # bool (H, W) from RMBG-2.0
    probe_flat_mask: np.ndarray,   # bool (n_gen,) from probe diff
    h_lat: int, w_lat: int,
    feather_sigma: float = 8.0,
) -> Image.Image:
    """
    Resize the generated object to fit the probe-detected bounding box,
    then paste it onto the scene with a feathered alpha blend.

    Interior of the object mask stays at alpha=1.0 (guaranteed by feather_mask).
    Returns the rough composite image.
    """
    H, W = scene.size[1], scene.size[0]
    px_probe = _token_to_pixel(probe_flat_mask, h_lat, w_lat, H, W)
    rows = np.where(px_probe.any(axis=1))[0]
    cols = np.where(px_probe.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        print("    [paste_object_at_probe] WARNING: empty probe mask -- skipping paste.")
        return scene

    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    bh, bw = y2 - y1, x2 - x1

    # Tight-crop the generated object using its RMBG mask
    r, c = np.where(obj_mask)
    if len(r) == 0:
        print("    [paste_object_at_probe] WARNING: empty RMBG mask -- skipping paste.")
        return scene
    oy1, oy2 = int(r.min()), int(r.max()) + 1
    ox1, ox2 = int(c.min()), int(c.max()) + 1
    obj_crop  = obj_img.crop((ox1, oy1, ox2, oy2))
    mask_crop = Image.fromarray(obj_mask[oy1:oy2, ox1:ox2].astype(np.uint8) * 255, "L")

    # Resize to probe bounding box
    obj_r  = obj_crop.resize((bw, bh), Image.LANCZOS)
    mask_r = mask_crop.resize((bw, bh), Image.BILINEAR)

    # Full-scene alpha with feathering (interior clamped to 1.0 by feather_mask)
    alpha_small = feather_mask(np.array(mask_r) > 127, feather_sigma)
    alpha_full  = np.zeros((H, W), dtype=float)
    alpha_full[y1:y2, x1:x2] = alpha_small
    alpha3 = alpha_full[:, :, np.newaxis]

    scene_f = np.array(scene).astype(float)
    paste_f = scene_f.copy()
    paste_f[y1:y2, x1:x2] = np.array(obj_r).astype(float)
    result = alpha3 * paste_f + (1.0 - alpha3) * scene_f
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ============================================================
# Stage B: scene insertion chain
# ============================================================

def run_sketch_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    generated: Dict[str, Tuple[Image.Image, np.ndarray]],
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    threshold: float,
    feather_sigma: float,
    out_dir: str,
    mode_tag: str,
    device: str,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Stage B: probe -> place -> refine (K/V inject) -> bg_replace.

    generated: {name: (obj_img, obj_mask)} from Stage A.
    mode_tag: short prefix for output filenames (e.g. 'lora' or 'cn').
    """
    imgs, obj_masks = [base], []
    all_obj_mask = np.zeros(h_lat * w_lat, dtype=bool)

    for i, edit in enumerate(edits):
        name     = edit["name"]
        prompt   = edit["prompt"]
        img_prev = imgs[-1]
        H, W     = img_prev.size[1], img_prev.size[0]

        if name not in generated:
            print(f"\n  [{mode_tag}] step {i+1}  {name}: no generated object found -- SKIPPING")
            imgs.append(img_prev)
            obj_masks.append(np.zeros(h_lat * w_lat, dtype=bool))
            continue

        obj_img, obj_mask = generated[name]
        print(f"\n  [{mode_tag}] step {i+1}/{len(edits)}  {name}")

        # 1. PROBE: find where in the scene the object should go
        print(f"    probe  ({probe_steps} steps) ...")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_probe.png"))

        target_raw   = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        base_stable  = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)
        target_clean = target_raw & ~base_stable
        obj_masks.append(target_clean)

        print(f"    target={target_clean.mean()*100:.1f}%")
        color_overlay(img_prev, target_clean, h_lat, w_lat,
                      color=(0, 180, 255), alpha=0.45).save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_target.png"))

        # 2. PLACE: paste generated object at probe-detected location
        rough_composite = paste_object_at_probe(
            img_prev, obj_img, obj_mask, target_clean,
            h_lat, w_lat, feather_sigma,
        )
        rough_composite.save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_rough.png"))

        # 3. REFINE: Kontext K/V inject from rough_composite
        #    - rough_composite as canvas: pasted object is a visual hint for Kontext
        #    - K/V injection locks background tokens to rough_composite features
        #    - Kontext smooths seams and adjusts object lighting
        print(f"    refine ({num_steps} steps, s={strength}, cutoff={cutoff}) ...")
        img_inject = run_kv_injected(
            pipe,
            canvas=rough_composite,
            prompt=prompt,
            background_mask=~target_clean,
            seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
            strength=strength, cutoff=cutoff,
            vital_layers=vital_layers,
            device=device,
        )
        img_inject.save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_inject.png"))

        # 4. BG-REPLACE: hard pixel-copy original base over all non-object tokens
        all_obj_mask = all_obj_mask | target_clean
        px_pure_bg   = _token_to_pixel(~all_obj_mask, h_lat, w_lat, H, W)
        inject_arr   = np.array(img_inject).astype(float)
        base_arr     = np.array(base).astype(float)
        inject_arr[px_pure_bg] = base_arr[px_pure_bg]
        img_curr = Image.fromarray(inject_arr.clip(0, 255).astype(np.uint8))
        img_curr.save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_result.png"))

        imgs.append(img_curr)

    return imgs, obj_masks


# ============================================================
# Visualisation
# ============================================================

def build_comparison_grid(
    imgs_by_mode: Dict[str, List[Image.Image]],
    edits: List[dict],
    out_dir: str,
    strength: float,
    feather_sigma: float,
):
    modes = list(imgs_by_mode.keys())
    ncols = len(edits) + 1
    labels = {
        "bg_replace": f"Text-only\nbg_replace (s={strength})",
        "sketch_lora": f"Sketch LoRA\n(σ={feather_sigma}px)",
        "sketch_cn": f"Sketch ControlNet\n(σ={feather_sigma}px)",
    }
    images, titles = [], []
    for mode in modes:
        lbl  = labels.get(mode, mode)
        imgs = imgs_by_mode[mode]
        images.append(imgs[0])
        titles.append(f"{lbl}\nbase")
        for j, edit in enumerate(edits):
            images.append(imgs[j + 1])
            titles.append(f"{lbl}\n{edit['name']}")

    save_grid(
        images, titles,
        os.path.join(out_dir, "KEY_RESULT_comparison.png"),
        ncols=ncols,
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sketch-conditioned multi-step object insertion via FLUX Kontext."
    )
    p.add_argument("--sketch_dir",       required=True,
                   help="Folder with {name}.png hand-drawn sketches.")
    p.add_argument("--hf_token",         required=True)
    p.add_argument("--cache_dir",        default="./models")
    p.add_argument("--out_dir",          default="results/phase1_sketch")
    p.add_argument("--config",           default=None,
                   help="JSON list of {name, prompt, description}. Overrides built-in EDITS.")
    p.add_argument("--mode",             default="lora",
                   choices=["lora", "controlnet", "all"],
                   help="'lora': Kontext+LoRA (single model). "
                        "'controlnet': FLUX ControlNet (stronger shape control). "
                        "'all': runs lora + controlnet + text-only bg_replace for comparison.")
    p.add_argument("--lora_id",          default="gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA")
    p.add_argument("--lora_guidance",    type=float, default=4.0,
                   help="guidance_scale for Stage A LoRA generation (4.0-5.0 recommended).")
    p.add_argument("--controlnet_id",    default="promeai/FLUX.1-controlnet-lineart-promeai",
                   help="Lineart ControlNet model (better for hand-drawn than Canny).")
    p.add_argument("--controlnet_scale", type=float, default=0.7,
                   help="ControlNet conditioning scale (0.5-0.8 for hand-drawn sketches).")
    p.add_argument("--rmbg_threshold",   type=float, default=0.5,
                   help="RMBG-2.0 alpha threshold for mask binarisation [0,1].")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_steps",        type=int,   default=28)
    p.add_argument("--probe_steps",      type=int,   default=18)
    p.add_argument("--guidance",         type=float, default=2.5,
                   help="guidance_scale for Stage B (scene insertion).")
    p.add_argument("--strength",         type=float, default=0.3,
                   help="K/V injection strength (Stage B).")
    p.add_argument("--cutoff",           type=float, default=0.4,
                   help="Inject first CUTOFF fraction of steps (Stage B).")
    p.add_argument("--threshold",        type=float, default=40.0,
                   help="Pixel diff threshold for probe mask detection.")
    p.add_argument("--feather_sigma",    type=float, default=8.0,
                   help="Gaussian blur radius (px) for object boundary feathering.")
    p.add_argument("--all_layers",       action="store_true",
                   help="Use all 57 layers for K/V injection (TIER_A is default).")
    p.add_argument("--height",           type=int,   default=1024)
    p.add_argument("--width",            type=int,   default=1024)
    p.add_argument("--device",           default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)
        print(f"Loaded {len(edits)} edits from {args.config}")

    h_lat = args.height // 16
    w_lat = args.width  // 16
    vital_layers = ALL_LAYERS if args.all_layers else TIER_A_LAYERS

    print(f"Edit sequence : {[e['name'] for e in edits]}")
    print(f"Mode          : {args.mode}")
    print(f"Sketch dir    : {args.sketch_dir}")
    print(f"feather_sigma : {args.feather_sigma} px")

    # ── 0. Load RMBG-2.0 (used by all modes for mask extraction) ─────────────
    print("\nLoading RMBG-2.0 for foreground mask extraction ...")
    rmbg_model = load_rmbg(args.cache_dir, args.device)

    # ── 1. Base scene ─────────────────────────────────────────────────────────
    # Base is generated once; shared across all modes for fair comparison.
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # We need Kontext even for base generation; load it now if starting with lora/all
    # For controlnet mode we may load Kontext after Stage A; still need it for base.
    print("Loading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    base = run_standard(pipe, grey, BASE_PROMPT,
                        args.seed, args.num_steps, args.guidance,
                        args.height, args.width)
    base.save(os.path.join(args.out_dir, "step0_base.png"))

    # ── 2. Run requested modes ─────────────────────────────────────────────────
    imgs_by_mode:      Dict[str, List[Image.Image]] = {}
    obj_masks_by_mode: Dict[str, List[np.ndarray]]  = {}

    run_lora       = args.mode in ("lora",  "all")
    run_controlnet = args.mode in ("controlnet", "all")
    run_text_only  = args.mode == "all"   # bg_replace baseline for comparison

    # ── Mode: LoRA ────────────────────────────────────────────────────────────
    if run_lora:
        generated = stage_a_lora(
            pipe=pipe,
            edits=edits,
            sketch_dir=args.sketch_dir,
            lora_id=args.lora_id,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance=args.lora_guidance,
            height=args.height, width=args.width,
            rmbg_model=rmbg_model,
            rmbg_threshold=args.rmbg_threshold,
            out_dir=args.out_dir,
            device=args.device,
        )
        print("\n=== Stage B: Scene Insertion (LoRA) ===")
        imgs, masks = run_sketch_chain(
            pipe=pipe, base=base, edits=edits, generated=generated,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
            guidance=args.guidance, height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers, threshold=args.threshold,
            feather_sigma=args.feather_sigma,
            out_dir=args.out_dir, mode_tag="lora", device=args.device,
        )
        imgs_by_mode["sketch_lora"]      = imgs
        obj_masks_by_mode["sketch_lora"] = masks
        imgs[-1].save(os.path.join(args.out_dir, "KEY_lora_final.png"))

    # ── Mode: ControlNet ──────────────────────────────────────────────────────
    if run_controlnet:
        # Offload Kontext if it is loaded, then reload after Stage A
        # (ControlNet FLUX.1-dev + Kontext both need large GPU memory)
        if pipe is not None:
            print("\nOffloading Kontext to free memory for ControlNet ...")
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            pipe = None

        generated_cn = stage_a_controlnet(
            edits=edits,
            sketch_dir=args.sketch_dir,
            controlnet_id=args.controlnet_id,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance=args.guidance,
            height=args.height, width=args.width,
            rmbg_model=rmbg_model,
            rmbg_threshold=args.rmbg_threshold,
            out_dir=args.out_dir,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            device=args.device,
        )
        # Reload Kontext for Stage B
        print("\nReloading FLUX.1-Kontext-dev for Stage B ...")
        pipe = load_kontext_pipeline(
            hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
        )
        print("\n=== Stage B: Scene Insertion (ControlNet) ===")
        imgs_cn, masks_cn = run_sketch_chain(
            pipe=pipe, base=base, edits=edits, generated=generated_cn,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
            guidance=args.guidance, height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers, threshold=args.threshold,
            feather_sigma=args.feather_sigma,
            out_dir=args.out_dir, mode_tag="cn", device=args.device,
        )
        imgs_by_mode["sketch_cn"]      = imgs_cn
        obj_masks_by_mode["sketch_cn"] = masks_cn
        imgs_cn[-1].save(os.path.join(args.out_dir, "KEY_controlnet_final.png"))

    # ── Mode: text-only bg_replace (comparison baseline) ─────────────────────
    if run_text_only:
        print("\n=== Text-only bg_replace (comparison baseline) ===")
        if pipe is None:
            pipe = load_kontext_pipeline(
                hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
            )
        imgs_br, masks_br = run_bg_replace_chain(
            pipe=pipe, base=base, edits=edits,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
            guidance=args.guidance, height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers, threshold=args.threshold,
            feather_sigma=args.feather_sigma,
            out_dir=args.out_dir, device=args.device,
        )
        imgs_by_mode["bg_replace"]      = imgs_br
        obj_masks_by_mode["bg_replace"] = masks_br
        imgs_br[-1].save(os.path.join(args.out_dir, "KEY_bg_replace_final.png"))

    # ── 3. Visualisation ──────────────────────────────────────────────────────
    if len(imgs_by_mode) > 1:
        print("\n=== Building comparison grid ===")
        build_comparison_grid(
            imgs_by_mode, edits, args.out_dir,
            strength=args.strength, feather_sigma=args.feather_sigma,
        )
        print("  Saved: KEY_RESULT_comparison.png")

    mask_modes = {m: v for m, v in obj_masks_by_mode.items()
                  if any(m_.any() for m_ in v)}
    if mask_modes and len(imgs_by_mode) > 1:
        print("=== Building stability grid ===")
        build_stability_grid(
            base, imgs_by_mode, edits, mask_modes,
            h_lat, w_lat, args.out_dir,
            strength=args.strength, feather_sigma=args.feather_sigma,
        )
        print("  Saved: KEY_RESULT_stability.png")

    # ── 4. What to check ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("WHAT TO CHECK")
    print(f"{'='*72}")
    print(f"""
sketch_gen_{{name}}.png + sketch_mask_{{name}}.png
  Left half: Stage A generated object. Right half: RMBG-2.0 mask.
  The mask should cleanly cover the object (including white objects like the vase).
  If mask leaks into background: raise --rmbg_threshold (try 0.65).
  If mask clips the object:      lower --rmbg_threshold (try 0.3).

sk_{{mode}}_step{{i}}_{{name}}_rough.png
  The generated object pasted at the probe-detected location.
  Verify: is the object at the right position and scale?
  If too small/large:   the probe bbox drives the size -- check probe output.
  If wrong position:    increase --probe_steps (try 22-24).

sk_{{mode}}_step{{i}}_{{name}}_inject.png
  After Kontext K/V injection. Seams should be smooth; lighting should match the room.
  If hard seams visible:  increase --feather_sigma (12-16).
  If seams gone but object looks different from sketch:  raise LoRA guidance or
    controlnet_scale.

KEY_lora_final.png vs KEY_bg_replace_final.png
  Does the sketch-mode object match the hand-drawn intent (shape/pose)?
  Does background look identical across all edit steps? (bg_replace guarantees this.)
""")


if __name__ == "__main__":
    main()
