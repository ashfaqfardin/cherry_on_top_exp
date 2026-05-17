"""
Experiment 2 — Semantic specialisation of layers.

For each of the 57 FLUX blocks, measure how much bypassing the block changes
the CLIP-space distance between semantically contrastive image pairs.

Speed note
----------
Full-model images for all pairs are pre-generated ONCE before the layer loop.
The original approach called generate_pair() inside the layer loop, paying the
full-model cost 57× per pair.  With pre-caching it is paid only once.

Quick-run defaults (fits ~15-20 min with cpu_offload):
  --n_pairs 1  --n_steps 4  --only_mm

Full sweep (all 57 layers, 10 pairs/category):
  --n_pairs 10  --n_steps 15  (no --only_mm)

Outputs
-------
results/semantic_sensitivity.npy   shape (n_layers, 7)
results/semantic_sensitivity.json

Usage
-----
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    [--n_pairs 1] [--n_steps 4] [--only_mm] \
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.layer_bypass import generate_with_bypass, load_pipeline

N_MM     = 19
N_SINGLE = 38
CATEGORIES = ["colour", "style", "material", "texture", "shape", "layout", "object"]


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def load_clip(device: str = "cuda"):
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


@torch.no_grad()
def clip_embed(img: Image.Image, model, processor, device: str) -> torch.Tensor:
    inputs = processor(images=img, return_tensors="pt").to(device)
    feats  = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1)


def clip_distance(img_a: Image.Image, img_b: Image.Image,
                  model, processor, device: str) -> float:
    fa = clip_embed(img_a, model, processor, device)
    fb = clip_embed(img_b, model, processor, device)
    return float(1.0 - F.cosine_similarity(fa, fb, dim=-1).item())


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    npy_path  = "results/semantic_sensitivity.npy"
    json_path = "results/semantic_sensitivity.json"

    with open("prompts/semantic_prompts.json") as f:
        semantic_prompts = json.load(f)

    block_list = [("mm", i) for i in range(N_MM)]
    if not args.only_mm:
        block_list += [("single", i) for i in range(N_SINGLE)]
    n_layers = len(block_list)

    # Load or initialise result matrix
    if os.path.exists(npy_path):
        matrix = np.load(npy_path)
        # Resize if layer count changed (e.g. only_mm resumed from full run)
        if matrix.shape[0] != n_layers:
            print(f"  Warning: saved matrix has {matrix.shape[0]} rows, expected {n_layers}. Starting fresh.")
            matrix = np.full((n_layers, len(CATEGORIES)), np.nan)
        else:
            print(f"Resuming from {npy_path}")
    else:
        matrix = np.full((n_layers, len(CATEGORIES)), np.nan)

    pipe       = load_pipeline(args.model_path, args.hf_token, args.device, args.cpu_offload)
    clip_model, clip_proc = load_clip(args.device)

    # ------------------------------------------------------------------
    # Pre-generate ALL full-model images for every pair we will use
    # Key: (cat, pair_idx, 'a'/'b') → PIL image
    # ------------------------------------------------------------------
    print("Pre-generating full-model reference images for all pairs...")
    full_cache: dict = {}
    for cat in CATEGORIES:
        pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
        for p_idx, (prompt_a, prompt_b) in enumerate(pairs):
            for side, prompt in [("a", prompt_a), ("b", prompt_b)]:
                cache_key = (cat, p_idx, side)
                if cache_key not in full_cache:
                    img = generate_with_bypass(
                        pipe, prompt, seed=42,
                        block_type="mm", bypass_idx=None,
                        num_inference_steps=args.n_steps, device=args.device,
                    )
                    full_cache[cache_key] = img
    n_cached = len(full_cache)
    print(f"  Cached {n_cached} full-model images.\n")

    # Pre-compute full-model CLIP distances (once)
    dist_full_cache: dict = {}
    for cat in CATEGORIES:
        pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
        for p_idx, _ in enumerate(pairs):
            img_a = full_cache[(cat, p_idx, "a")]
            img_b = full_cache[(cat, p_idx, "b")]
            dist_full_cache[(cat, p_idx)] = clip_distance(
                img_a, img_b, clip_model, clip_proc, args.device
            )

    # ------------------------------------------------------------------
    # Layer sweep — only generate ablated images
    # ------------------------------------------------------------------
    for global_idx, (block_type, layer_idx) in enumerate(block_list):
        if not np.any(np.isnan(matrix[global_idx])):
            tag = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
            print(f"  [{global_idx:2d}] {tag}: cached")
            continue

        tag = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
        print(f"  [{global_idx:2d}] {tag} ...", flush=True)

        for cat_idx, cat in enumerate(CATEGORIES):
            if not np.isnan(matrix[global_idx, cat_idx]):
                continue

            pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
            deltas = []
            for p_idx, (prompt_a, prompt_b) in enumerate(pairs):
                # Only generate ablated images — full images come from cache
                img_a_abl = generate_with_bypass(
                    pipe, prompt_a, seed=42,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )
                img_b_abl = generate_with_bypass(
                    pipe, prompt_b, seed=42,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )
                dist_abl = clip_distance(img_a_abl, img_b_abl, clip_model, clip_proc, args.device)
                dist_full = dist_full_cache[(cat, p_idx)]
                deltas.append(abs(dist_full - dist_abl))

            matrix[global_idx, cat_idx] = float(np.mean(deltas)) if deltas else 0.0

        print(f"     {dict(zip(CATEGORIES, [round(float(x), 4) for x in matrix[global_idx]]))}")
        np.save(npy_path, matrix)

    # Save JSON
    layer_labels = (
        [f"MM-{i}" for i in range(N_MM)] +
        ([f"S-{i}" for i in range(N_SINGLE)] if not args.only_mm else [])
    )
    result_dict = {
        "categories":  CATEGORIES,
        "layers":      layer_labels[:n_layers],
        "matrix":      matrix.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nDone. Saved:\n  {npy_path}\n  {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--n_pairs",    type=int, default=1,
                        help="Contrastive pairs per category (default 1; max 50)")
    parser.add_argument("--n_steps",    type=int, default=4,
                        help="Denoising steps (default 4; paper uses 15)")
    parser.add_argument("--only_mm",    action="store_true",
                        help="Only sweep MM-DiT blocks (19 layers) — faster")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
