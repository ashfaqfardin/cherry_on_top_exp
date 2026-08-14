# -*- coding: utf-8 -*-
"""
phase1_ref_qwen.py — No-mask pipeline: native multi-image reference conditioning

Uses Qwen-Image-Edit-2509's built-in multi-image support (image concatenation
training) to pass scene + object reference as separate images — no canvas
stitching, no crop, no bleed artifacts.

Pipeline per insertion step
----------------------------
  1. Stage A  : sketch → obj_img  (Qwen renders the sketch)
  2. Stage K  : Qwen(image=[scene, obj_img], prompt=place_prompt) → 1024×1024

Pipeline per removal step
--------------------------
  1. Qwen(image=scene, prompt=remove_prompt) → next scene

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
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image


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
    pipe,
    image,           # PIL Image or list of PIL Images
    prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    negative_prompt: str = "blurry, distorted, low quality, watermark, text, artifacts",
) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=image, num_inference_steps=num_steps,
        true_cfg_scale=guidance, guidance_scale=1.0,
        height=height, width=width,
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
        f"Render this hand-drawn sketch as a photorealistic {description}. "
        f"Place it centered on a plain white background. "
        f"Studio lighting, no shadows, no background objects, high detail."
    )
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts, background, room, floor",
        image=sketch, num_inference_steps=num_steps,
        true_cfg_scale=guidance, guidance_scale=1.0,
        height=height, width=width,
        generator=gen,
    ).images[0]


# ── Prompts ────────────────────────────────────────────────────────────────────

def insert_prompt(description: str) -> str:
    # The pipeline prepends "Picture 1: <img> Picture 2: <img>" to this prompt,
    # so reference images by those labels. Avoid specifying center placement.
    return (
        f"Picture 1 shows a room. Picture 2 shows a {description}. "
        f"Edit Picture 1: place the {description} from Picture 2 into the room "
        f"— position it naturally to one side of the room, resting on the wooden floor "
        f"with a realistic contact shadow and lighting consistent with the room. "
        f"Do not place it in the dead center of the image. "
        f"Keep all existing objects and all other parts of the room exactly as they are."
    )


def remove_prompt(description: str) -> str:
    return (
        f"Remove the {description} from the room completely. "
        f"Fill the vacated area with seamless wooden floor and white wall that match "
        f"the surrounding surfaces exactly. "
        f"Keep all other objects and all other parts of the room exactly as they are."
    )


# ── Main chain ────────────────────────────────────────────────────────────────

def run_chain(
    pipe,
    base:         Image.Image,
    edits:        List[dict],
    sketch_dir:   str,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    obj_guidance: float,
    height:       int,
    width:        int,
    out_dir:      str,
) -> List[Image.Image]:
    results   = [base]
    scene     = base
    obj_cache: dict = {}

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")
        tag    = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        if action == "insert":
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

            qwen_image = [scene.convert("RGB"), obj_img.convert("RGB")]
            prompt = insert_prompt(desc)
            neg_p  = "blurry, distorted, low quality, watermark, text, artifacts"
            print(f"  [K] Multi-image pass: [scene, obj_ref] → {width}×{height}")

        else:  # remove
            qwen_image = scene.convert("RGB")
            prompt = remove_prompt(desc)
            neg_p  = f"blurry, distorted, low quality, watermark, text, artifacts, {desc}"
            print(f"  [K] Removal pass → {width}×{height}")

        with open(os.path.join(out_dir, f"prompt_{tag}.txt"), "w") as f:
            f.write(prompt)
        print(f"      {prompt[:120]}...")

        next_scene = run_qwen(
            pipe=pipe, image=qwen_image, prompt=prompt,
            seed=seed, num_steps=num_steps, guidance=guidance,
            height=height, width=width, negative_prompt=neg_p,
        )
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
    p.add_argument("--guidance",     type=float, default=4.0)
    p.add_argument("--obj_guidance", type=float, default=4.0)
    p.add_argument("--num_steps",    type=int,   default=50)
    p.add_argument("--height",       type=int,   default=1024)
    p.add_argument("--width",        type=int,   default=1024)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--cpu_offload",  action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{_SEP}")
    print(f"  phase1_ref_qwen  —  No-mask multi-image reference pipeline")
    print(f"{_SEP}")
    print(f"  Model      : {args.model}")
    print(f"  Sketch dir : {args.sketch_dir}")
    print(f"  Output     : {args.width}×{args.height}  (scene + obj_ref passed as separate images)")
    print(f"  Guidance   : scene={args.guidance}  obj={args.obj_guidance}")
    print(f"  Steps      : {args.num_steps}")
    print(f"  Out dir    : {args.out_dir}")
    print(f"{_SEP}\n")

    print(f"Loading {args.model} ...")
    pipe = load_pipeline(
        model=args.model, hf_token=args.hf_token,
        cache_dir=args.cache_dir, device=args.device,
        cpu_offload=args.cpu_offload,
    )

    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_qwen(
        pipe=pipe, image=grey, prompt=BASE_PROMPT,
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
