# -*- coding: utf-8 -*-
"""
phase1_ref_qwen.py — No-mask pipeline: side-by-side reference conditioning

Simplified alternative to phase1_collage_qwen.py.
No placement masks, no manual paste, no BLD callbacks, no K/V injection.

Pipeline per insertion step
----------------------------
  1. Stage A  : sketch → obj_img  (Qwen renders the sketch)
  2. Canvas   : [scene (75%) | obj_img reference (25%)] → 1024×1024
  3. Stage K  : Qwen(image=canvas, prompt=place_prompt) → 1024×1024 output
  4. Crop     : take left 75% of output → resize to 1024×1024 → next scene

Pipeline per removal step
--------------------------
  1. Qwen(image=scene, prompt=remove_prompt) → next scene
  (no reference panel needed — instruction-following handles it)

Trade-offs vs the masked pipeline
-----------------------------------
  + No mask files required
  + Natural shadows / lighting from Qwen natively
  + Much simpler code
  - Placement is Qwen's choice, not yours
  - Object identity may drift (Qwen interprets the reference loosely)
  - Incremental: each step sees prior edits, Qwen may recompose

Usage
------
  python NewWork/KontextEval/phase1_ref_qwen.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token   $HF_TOKEN \\
      --cache_dir  ./models \\
      --out_dir    results/phase1_ref_qwen
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image, ImageDraw, ImageFont


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]

BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)

_SEP = "═" * 60


# ── Pipeline loading ───────────────────────────────────────────────────────────

def load_pipeline(
    model: str = "Qwen/Qwen-Image-Edit-2509",
    hf_token: str | None = None,
    cache_dir: str = "./models",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model, torch_dtype=dtype, token=hf_token, cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Qwen call ─────────────────────────────────────────────────────────────────

def run_qwen(
    pipe, canvas: Image.Image, prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    negative_prompt: str = "blurry, distorted, low quality, watermark, text, artifacts",
) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=canvas, num_inference_steps=num_steps,
        true_cfg_scale=guidance, height=height, width=width,
        generator=gen,
    ).images[0]


# ── Stage A: sketch → object image ────────────────────────────────────────────

def generate_object(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
) -> Image.Image:
    sketch = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    prompt = (
        f"Render this sketch as a photorealistic {description} "
        "on a plain white background. No shadows. Studio lighting. High quality."
    )
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
        image=sketch, num_inference_steps=num_steps,
        true_cfg_scale=guidance, height=height, width=width,
        generator=gen,
    ).images[0]


# ── Side-by-side canvas builder ────────────────────────────────────────────────

def build_reference_canvas(
    scene:      Image.Image,
    obj_img:    Image.Image,
    height:     int = 1024,
    width:      int = 1024,
    scene_frac: float = 0.75,
    label:      str = "",
) -> Image.Image:
    """
    Build a [scene | reference] canvas at (width × height).

    scene_frac=0.75 → left 768px = scene, right 256px = object reference.
    A thin grey divider separates the panels. An optional label is rendered
    in the reference panel to help Qwen understand the layout.
    """
    scene_w = int(width * scene_frac)
    ref_w   = width - scene_w

    canvas = Image.new("RGB", (width, height), (240, 240, 240))

    # Left: scene
    scene_rs = scene.convert("RGB").resize((scene_w, height), Image.LANCZOS)
    canvas.paste(scene_rs, (0, 0))

    # Divider
    draw = ImageDraw.Draw(canvas)
    draw.line([(scene_w, 0), (scene_w, height)], fill=(160, 160, 160), width=3)

    # Right: object reference — fit inside ref panel with padding
    pad   = 16
    inner = ref_w - 2 * pad
    obj_rs = obj_img.convert("RGB")
    obj_rs.thumbnail((inner, height - 2 * pad), Image.LANCZOS)
    ox = scene_w + pad + (inner - obj_rs.width)  // 2
    oy = pad + (height - 2 * pad - obj_rs.height) // 2
    canvas.paste(obj_rs, (ox, oy))

    # Label at bottom of reference panel
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        draw.text((scene_w + pad, height - 36), f"REF: {label}", fill=(80, 80, 80), font=font)

    return canvas


# ── Crop left panel and restore to full resolution ─────────────────────────────

def crop_scene(
    output:     Image.Image,
    height:     int = 1024,
    width:      int = 1024,
    scene_frac: float = 0.75,
) -> Image.Image:
    """
    Crop the left scene_frac of the output and resize back to (width × height).
    The reference panel is discarded.
    """
    scene_w = int(output.width * scene_frac)
    cropped = output.crop((0, 0, scene_w, output.height))
    return cropped.resize((width, height), Image.LANCZOS)


# ── Insertion prompt ───────────────────────────────────────────────────────────

def insert_prompt(description: str) -> str:
    return (
        f"This image has two sections separated by a vertical grey line. "
        f"The LEFT section (left three-quarters) shows a room. "
        f"The RIGHT section (right quarter) shows a reference {description}. "
        f"Edit the LEFT section ONLY: place the {description} from the right reference "
        f"naturally in the room — resting on the wooden floor with realistic shadows and "
        f"contact lighting that match the room. "
        f"Do NOT modify the right reference panel. "
        f"Keep all other parts of the room exactly as they are."
    )


# ── Removal prompt ─────────────────────────────────────────────────────────────

def remove_prompt(description: str) -> str:
    return (
        f"Remove the {description} from the room completely. "
        f"Fill the area with seamless wooden floor and white wall that match the "
        f"surrounding surfaces. Keep all other objects and parts of the room exactly "
        f"the same."
    )


# ── Main chain ────────────────────────────────────────────────────────────────

def run_chain(
    pipe,
    base:       Image.Image,
    edits:      List[dict],
    sketch_dir: str,
    seed:       int,
    num_steps:  int,
    guidance:   float,
    obj_guidance: float,
    height:     int,
    width:      int,
    scene_frac: float,
    out_dir:    str,
) -> List[Image.Image]:
    results  = [base]
    scene    = base
    obj_cache: dict = {}   # name → obj_img  (reuse across steps)

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")
        tag    = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        if action == "insert":
            # Stage A — generate object image (cached)
            if name not in obj_cache:
                sketch_path = os.path.join(sketch_dir, f"{name}.png")
                if not os.path.isfile(sketch_path):
                    sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
                if not os.path.isfile(sketch_path):
                    raise FileNotFoundError(
                        f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                        f"in {sketch_dir!r}"
                    )
                print(f"  [A] Generating '{desc}' from sketch ...")
                obj_img = generate_object(
                    pipe=pipe, sketch_path=sketch_path, description=desc,
                    seed=seed, num_steps=num_steps, guidance=obj_guidance,
                    height=height, width=width,
                )
                obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))
                obj_cache[name] = obj_img
                print(f"      Saved: obj_gen_{name}.png")
            else:
                print(f"  [A] Reusing cached obj_gen_{name}.png")
                obj_img = obj_cache[name]

            # Build side-by-side canvas
            canvas = build_reference_canvas(
                scene=scene, obj_img=obj_img,
                height=height, width=width,
                scene_frac=scene_frac, label=desc,
            )
            canvas.save(os.path.join(out_dir, f"canvas_{tag}.png"))
            print(f"  [CANVAS] Saved: canvas_{tag}.png  "
                  f"(scene={int(width*scene_frac)}px | ref={width-int(width*scene_frac)}px)")

            prompt   = insert_prompt(desc)
            neg_p    = "blurry, distorted, low quality, watermark, text, artifacts"

        else:  # remove
            canvas = scene   # no reference panel needed
            prompt = remove_prompt(desc)
            neg_p  = f"blurry, distorted, low quality, watermark, text, artifacts, {desc}"
            print(f"  [A] Removal — using scene directly (no reference panel)")

        # Save prompt
        with open(os.path.join(out_dir, f"prompt_{tag}.txt"), "w") as f:
            f.write(prompt)
        print(f"  [K] Qwen pass  ({num_steps} steps, cfg={guidance}) ...")
        print(f"      Prompt: {prompt[:120]}...")

        raw_output = run_qwen(
            pipe=pipe, canvas=canvas, prompt=prompt,
            seed=seed, num_steps=num_steps, guidance=guidance,
            height=height, width=width, negative_prompt=neg_p,
        )
        raw_output.save(os.path.join(out_dir, f"raw_{tag}.png"))

        if action == "insert":
            # Crop left panel → resize to full resolution
            next_scene = crop_scene(raw_output, height=height, width=width,
                                    scene_frac=scene_frac)
            print(f"  [CROP] Cropped left {scene_frac:.0%} → resized to {width}×{height}")
        else:
            next_scene = raw_output

        next_scene.save(os.path.join(out_dir, f"result_{tag}.png"))
        print(f"      Saved: result_{tag}.png")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Grid helper ────────────────────────────────────────────────────────────────

def save_grid(images, titles, path, ncols=None, figsize_per=(4, 4)):
    n     = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="No-mask reference pipeline for Qwen-Image-Edit")
    p.add_argument("--sketch_dir",   required=True)
    p.add_argument("--hf_token",     required=True)
    p.add_argument("--cache_dir",    default="./models")
    p.add_argument("--out_dir",      default="results/phase1_ref_qwen")
    p.add_argument("--model",        default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--guidance",     type=float, default=3.5)
    p.add_argument("--obj_guidance", type=float, default=3.5)
    p.add_argument("--num_steps",    type=int,   default=50)
    p.add_argument("--height",       type=int,   default=1024)
    p.add_argument("--width",        type=int,   default=1024)
    p.add_argument("--scene_frac",   type=float, default=0.75,
                   help="Fraction of canvas width given to the scene (default 0.75 = 768px).")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--cpu_offload",  action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{_SEP}")
    print(f"  phase1_ref_qwen  —  No-mask reference pipeline")
    print(f"{_SEP}")
    print(f"  Model      : {args.model}")
    print(f"  Sketch dir : {args.sketch_dir}")
    print(f"  Scene frac : {args.scene_frac}  ({int(args.width*args.scene_frac)}px scene "
          f"| {args.width-int(args.width*args.scene_frac)}px reference)")
    print(f"  Guidance   : scene={args.guidance}  obj={args.obj_guidance}")
    print(f"  Steps      : {args.num_steps}")
    print(f"  Output     : {args.out_dir}")
    print(f"{_SEP}\n")

    print(f"Loading {args.model} ...")
    pipe = load_pipeline(
        model=args.model, hf_token=args.hf_token,
        cache_dir=args.cache_dir, device=args.device,
        cpu_offload=args.cpu_offload,
    )

    # Base scene
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_qwen(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    results = run_chain(
        pipe=pipe, base=base, edits=EDITS,
        sketch_dir=args.sketch_dir,
        seed=args.seed, num_steps=args.num_steps,
        guidance=args.guidance, obj_guidance=args.obj_guidance,
        height=args.height, width=args.width,
        scene_frac=args.scene_frac,
        out_dir=args.out_dir,
    )

    labels = ["base"] + [
        f"{'−' if e['action']=='remove' else '+'}{e['name']}" for e in EDITS
    ]
    save_grid(results, labels,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(results))

    print(f"\n{_SEP}")
    print(f"  Done. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
