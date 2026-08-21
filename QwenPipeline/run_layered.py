"""
run_layered.py — Incremental object insertion + RGBA layer decomposition.

Extends run_baseline.py with QwenImageLayeredPipeline (arxiv:2512.15603):
  After every insertion step, the updated scene is decomposed into N+1 RGBA
  layers (background + one per inserted object so far).

  Each layer is an RGBA PIL image — the alpha channel gives a clean object
  mask with no manual thresholding. These masks are saved alongside the flat
  scene composites and can be used directly for downstream editing.

Pipeline:
  [Edit pipe]   Si → Oi          (sketch → realistic object, same as baseline)
  [Edit pipe]   grey → Bi        (base room generation)
  [Edit pipe]   Bn + On → B(n+1) (incremental flat insertion)
  [Layer pipe]  B(n+1) → RGBA layers  (decompose into n+2 transparent layers)

Memory strategy: both pipelines use enable_model_cpu_offload() so only the
active model's layers sit on GPU at any moment.
"""

import gc
import os
import argparse
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageLayeredPipeline


# ── Edit list ─────────────────────────────────────────────────────────────────

EDITS = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "black ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
    {"name": "chair",   "description": "wooden chair"},
    {"name": "lamp",    "description": "modern lamp"},
    {"name": "plant",   "description": "potted green plant"},
    {"name": "backpack", "description": "blue backpack"},
]

BASE_PROMPT = (
    "A photorealistic empty room with a wooden floor, white walls, "
    "and a window letting in natural light. No objects on the floor."
)


# ── Edit pipeline steps (same as baseline) ────────────────────────────────────

def sketch_to_object(pipe, sketch: Image.Image, description: str, args) -> Image.Image:
    prompt = (
        f"Convert this sketch into a photorealistic {description} "
        f"on a plain white background. Keep the object centered and well-lit."
    )
    with torch.inference_mode():
        return pipe(
            image=[sketch],
            prompt=prompt,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="blurry, distorted, low quality, extra objects, background clutter",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


def generate_base(pipe, args) -> Image.Image:
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    with torch.inference_mode():
        return pipe(
            image=[grey],
            prompt=BASE_PROMPT,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="objects on floor, furniture, cluttered, dark",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


def add_object(pipe, scene: Image.Image, obj: Image.Image, description: str, args) -> Image.Image:
    prompt = (
        f"The first image is a room scene. The second image shows a {description}. "
        f"Place the {description} naturally on the floor of the room. "
        f"Preserve the room's lighting, perspective, and all existing contents exactly."
    )
    with torch.inference_mode():
        return pipe(
            image=[scene, obj],
            prompt=prompt,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="blurry, distorted, duplicate objects, wrong perspective",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


# ── Layer decomposition step (new) ────────────────────────────────────────────

def decompose_layers(
    layer_pipe,
    scene: Image.Image,
    n_layers: int,
    args,
) -> list:
    """
    Decompose scene into n_layers RGBA images using QwenImageLayeredPipeline.

    Returns a list of n_layers PIL RGBA images ordered back-to-front:
      index 0 = background layer
      index 1..n = object layers (one per inserted object so far)

    Alpha channel of each layer is the object mask — 255 where the layer
    contributes, 0 where it is transparent.
    """
    scene_rgba = scene.convert("RGBA")
    with torch.inference_mode():
        output = layer_pipe(
            image=scene_rgba,
            layers=n_layers,
            num_inference_steps=args.layer_steps,
            generator=torch.manual_seed(args.seed),
        )
    return output.images   # list of RGBA PIL Images


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_sketch(sketch_dir: str, name: str) -> str:
    for fname in (f"{name}.png", f"sketch_{name}.png", f"{name}.jpg"):
        p = os.path.join(sketch_dir, fname)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Sketch not found for '{name}' in {sketch_dir!r}")


def save_grid(images: list, labels: list, path: str):
    W, H = images[0].size
    n    = len(images)
    grid = Image.new("RGB", (W * n, H + 30), (240, 240, 240))
    for i, img in enumerate(images):
        grid.paste(img.convert("RGB"), (i * W, 30))
    grid.save(path)


def alpha_to_mask(rgba: Image.Image) -> Image.Image:
    """Extract alpha channel as a greyscale mask image."""
    return rgba.split()[3]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sketch_dir",      required=True)
    p.add_argument("--out_dir",         default="results/qwen_layered")
    p.add_argument("--edit_model_id",   default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--layer_model_id",  default="Qwen/Qwen-Image-Layered")
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--num_steps",       type=int,   default=40)
    p.add_argument("--layer_steps",     type=int,   default=50)
    p.add_argument("--true_cfg_scale",  type=float, default=4.0)
    p.add_argument("--guidance_scale",  default=None, type=float)
    p.add_argument("--height",          type=int,   default=1024)
    p.add_argument("--width",           type=int,   default=1024)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--skip_layered",    action="store_true",
                   help="Skip layer decomposition — run edit steps only (faster debug).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load edit pipeline ─────────────────────────────────────────────────
    print(f"Loading edit pipeline: {args.edit_model_id} ...")
    edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.edit_model_id, torch_dtype=torch.bfloat16
    )
    edit_pipe.enable_model_cpu_offload()
    edit_pipe.set_progress_bar_config(disable=None)
    print("Edit pipeline ready.\n")

    results      = []
    inserted_names = []

    # ── Step 1: Sketch → Object ────────────────────────────────────────────
    objects = {}
    for edit in EDITS:
        name = edit["name"]
        desc = edit["description"]
        sketch_path = find_sketch(args.sketch_dir, name)
        sketch = Image.open(sketch_path).convert("RGB").resize((args.width, args.height))
        print(f"[S→O] {name}: converting sketch ...")
        obj_img = sketch_to_object(edit_pipe, sketch, desc, args)
        obj_img.save(os.path.join(args.out_dir, f"obj_{name}.png"))
        objects[name] = obj_img
        print(f"      Saved obj_{name}.png")

    # ── Step 2: Base scene ─────────────────────────────────────────────────
    print("\n[BASE] Generating base room scene ...")
    base = generate_base(edit_pipe, args)
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    results.append(base)
    print("       Saved base_scene.png")

    # ── Step 3+4: Incremental insertion + layer decomposition ──────────────
    # Offload edit pipeline before loading layer pipeline later
    scene = base
    scenes_for_decomposition = []   # collect flat scenes for layer pass

    for i, edit in enumerate(EDITS):
        name = edit["name"]
        desc = edit["description"]
        print(f"\n[ADD] Step {i+1}/{len(EDITS)}: inserting '{desc}' ...")
        scene = add_object(edit_pipe, scene, objects[name], desc, args)
        scene.save(os.path.join(args.out_dir, f"scene_step{i+1}_{name}.png"))
        results.append(scene)
        inserted_names.append(name)
        scenes_for_decomposition.append((i + 1, name, scene.copy()))
        print(f"      Saved scene_step{i+1}_{name}.png")

    # ── Save chain grid (flat composites) ─────────────────────────────────
    labels = ["base"] + [e["name"] for e in EDITS]
    save_grid(results, labels, os.path.join(args.out_dir, "chain_grid.png"))
    print(f"\nChain grid saved.")

    if args.skip_layered:
        print("--skip_layered set, done.")
        return

    # ── Load layer pipeline ────────────────────────────────────────────────
    # Free edit pipeline memory before loading second model
    del edit_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nLoading layer pipeline: {args.layer_model_id} ...")
    layer_pipe = QwenImageLayeredPipeline.from_pretrained(
        args.layer_model_id, torch_dtype=torch.bfloat16
    )
    layer_pipe.enable_model_cpu_offload()
    layer_pipe.set_progress_bar_config(disable=None)
    print("Layer pipeline ready.\n")

    # ── Decompose each scene into RGBA layers ──────────────────────────────
    # n_layers = background (1) + number of objects inserted so far
    for step_idx, name, flat_scene in scenes_for_decomposition:
        n_layers = step_idx + 1   # background + step_idx objects
        print(f"[LAYER] Decomposing scene_step{step_idx}_{name} into {n_layers} layers ...")

        layers = decompose_layers(layer_pipe, flat_scene, n_layers, args)

        step_dir = os.path.join(args.out_dir, f"layers_step{step_idx}_{name}")
        os.makedirs(step_dir, exist_ok=True)

        layer_labels = ["background"] + inserted_names[:step_idx]
        for j, (layer_img, label) in enumerate(zip(layers, layer_labels)):
            layer_img.save(os.path.join(step_dir, f"layer_{j:02d}_{label}.png"))
            mask = alpha_to_mask(layer_img)
            mask.save(os.path.join(step_dir, f"mask_{j:02d}_{label}.png"))

        # Grid of RGB views of all layers for this step
        save_grid(layers, layer_labels,
                  os.path.join(step_dir, "layers_grid.png"))
        print(f"      Saved {len(layers)} layers → {step_dir}/")

    print(f"\nDone. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
