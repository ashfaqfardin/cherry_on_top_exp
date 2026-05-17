"""
Experiment 1 — Layer vitality sweep.

For each of the 57 FLUX blocks (19 MM-DiT + 38 single-stream), bypass the
block during generation and measure how much the output changes via DINOv2
cosine similarity.  Low similarity → high vitality.

Speed note
----------
Reference images are pre-generated ONCE before the sweep begins — the original
approach regenerated them inside the layer loop (×57 wasteful).  This alone
cuts total image count from  N_PROMPTS × (1+N_SEEDS) × 57  down to
N_PROMPTS + N_PROMPTS × N_SEEDS × 57.

Quick-run defaults (fits ~30-40 min with cpu_offload):
  --n_prompts 4   --n_seeds 1   --n_steps 4

Full-paper reproduction:
  --n_prompts 64  --n_seeds 4   --n_steps 15

Outputs
-------
results/vitality_scores.json

Usage
-----
python experiments/vitality_sweep.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    [--n_prompts 4] [--n_seeds 1] [--n_steps 4] \
    [--cpu_offload] [--device cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.layer_bypass import generate_with_bypass, load_pipeline

N_MM     = 19
N_SINGLE = 38

DINO_PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# DINOv2 helpers
# ---------------------------------------------------------------------------

def load_dino(device: str = "cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def dino_similarity(img_a: Image.Image, img_b: Image.Image, dino_model,
                    device: str = "cuda") -> float:
    ta = DINO_PREPROCESS(img_a).unsqueeze(0).to(device)
    tb = DINO_PREPROCESS(img_b).unsqueeze(0).to(device)
    fa = dino_model(ta)
    fb = dino_model(tb)
    return float(F.cosine_similarity(fa, fb, dim=-1).item())


# ---------------------------------------------------------------------------
# Main sweep  (references pre-cached outside the layer loop)
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    output_path = "results/vitality_scores.json"

    # Resume from partial results
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming from {output_path}")
    else:
        results = {"mm": {}, "single": {}}

    with open("prompts/vitality_prompts.json") as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.n_prompts]
    seeds   = list(range(args.n_seeds))

    n_total = (len(results["mm"]) + len(results["single"]))
    n_remaining = (N_MM + N_SINGLE) - n_total
    n_images_remaining = n_remaining * len(prompts) * len(seeds)
    print(f"Prompts: {len(prompts)}  Seeds: {len(seeds)}  Steps: {args.n_steps}")
    print(f"Layers remaining: {n_remaining} / {N_MM + N_SINGLE}")
    print(f"Estimated images to generate: {n_images_remaining}")
    print(f"  (references pre-cached once = {len(prompts)} extra images)\n")

    pipe = load_pipeline(args.model_path, args.hf_token, args.device,
                         args.cpu_offload)
    dino = load_dino(args.device)

    # ------------------------------------------------------------------
    # Pre-generate reference images (full model, seed 0) — done ONCE
    # ------------------------------------------------------------------
    print("Pre-generating reference images (full model, seed 0)...")
    ref_images: list[Image.Image] = []
    for i, prompt in enumerate(prompts):
        img = generate_with_bypass(
            pipe, prompt, seed=0,
            block_type="mm", bypass_idx=None,
            num_inference_steps=args.n_steps, device=args.device,
        )
        ref_images.append(img)
        print(f"  ref {i+1}/{len(prompts)} done")

    # ------------------------------------------------------------------
    # Sweep all blocks, reusing the cached references
    # ------------------------------------------------------------------
    block_list = (
        [("mm",     i) for i in range(N_MM)]    +
        [("single", i) for i in range(N_SINGLE)]
    )

    for block_type, layer_idx in block_list:
        key = str(layer_idx)
        store = results[block_type if block_type == "mm" else "single"]
        if key in store:
            tag = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
            print(f"  {tag}: cached, skipping")
            continue

        tag = f"MM-{layer_idx:2d}" if block_type == "mm" else f"S-{layer_idx:2d}"
        print(f"  {tag} ...", end=" ", flush=True)

        scores = []
        for i, prompt in enumerate(prompts):
            ref_img = ref_images[i]          # reuse — no regeneration
            for seed in seeds:
                ablated = generate_with_bypass(
                    pipe, prompt, seed=seed,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )
                scores.append(dino_similarity(ref_img, ablated, dino, args.device))

        bucket = "mm" if block_type == "mm" else "single"
        results[bucket][key] = scores
        print(f"mean sim = {float(np.mean(scores)):.4f}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX layer vitality sweep")
    parser.add_argument("--model_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",  type=str, required=True)
    parser.add_argument("--n_prompts", type=int, default=4,
                        help="Prompts to use (default 4; paper uses 64)")
    parser.add_argument("--n_seeds",   type=int, default=1,
                        help="Seeds per prompt (default 1; paper uses 4)")
    parser.add_argument("--n_steps",   type=int, default=4,
                        help="Denoising steps (default 4; paper uses 15)")
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
