"""
Experiment 2 — Semantic specialisation of layers.

For each of the 57 FLUX blocks, measure how much bypassing the block changes
the CLIP-space distance between semantically contrastive image pairs.

High sensitivity in category C → the layer encodes semantic C.

Outputs
-------
results/semantic_sensitivity.npy   shape (57, 7)
results/semantic_sensitivity.json  same data, JSON

Rows 0-18 = MM-DiT blocks; rows 19-56 = single-stream blocks.
Columns = colour, style, material, texture, shape, layout, object.

Usage
-----
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    [--n_pairs 10] [--n_steps 15] [--cpu_offload] [--device cuda]
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
def clip_image_embedding(img: Image.Image, model, processor, device: str) -> torch.Tensor:
    inputs = processor(images=img, return_tensors="pt").to(device)
    feats  = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1)


def clip_distance(img_a: Image.Image, img_b: Image.Image,
                  model, processor, device: str) -> float:
    fa = clip_image_embedding(img_a, model, processor, device)
    fb = clip_image_embedding(img_b, model, processor, device)
    # cosine similarity → distance = 1 - sim
    return float(1.0 - F.cosine_similarity(fa, fb, dim=-1).item())


# ---------------------------------------------------------------------------
# Sensitivity for a single layer
# ---------------------------------------------------------------------------

def semantic_sensitivity(
    layer_idx: int,
    block_type: str,
    prompt_pair: tuple,
    pipe,
    clip_model,
    clip_processor,
    device: str,
    n_steps: int,
    seed: int = 42,
) -> float:
    """
    Return |dist_full - dist_ablated| for one contrastive pair.
    """
    prompt_a, prompt_b = prompt_pair

    img_a_full = generate_with_bypass(pipe, prompt_a, seed=seed,
                                      block_type=block_type, bypass_idx=None,
                                      num_inference_steps=n_steps, device=device)
    img_b_full = generate_with_bypass(pipe, prompt_b, seed=seed,
                                      block_type=block_type, bypass_idx=None,
                                      num_inference_steps=n_steps, device=device)

    img_a_abl = generate_with_bypass(pipe, prompt_a, seed=seed,
                                     block_type=block_type, bypass_idx=layer_idx,
                                     num_inference_steps=n_steps, device=device)
    img_b_abl = generate_with_bypass(pipe, prompt_b, seed=seed,
                                     block_type=block_type, bypass_idx=layer_idx,
                                     num_inference_steps=n_steps, device=device)

    dist_full = clip_distance(img_a_full, img_b_full, clip_model, clip_processor, device)
    dist_abl  = clip_distance(img_a_abl,  img_b_abl,  clip_model, clip_processor, device)

    return abs(dist_full - dist_abl)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    npy_path  = "results/semantic_sensitivity.npy"
    json_path = "results/semantic_sensitivity.json"

    with open("prompts/semantic_prompts.json") as f:
        semantic_prompts = json.load(f)

    pipe = load_pipeline(args.model_path, args.hf_token, args.device,
                         args.cpu_offload)
    clip_model, clip_processor = load_clip(args.device)

    # Load or initialise result matrix (57, 7)
    n_layers = N_MM + N_SINGLE
    n_cats   = len(CATEGORIES)
    if os.path.exists(npy_path):
        matrix = np.load(npy_path)
        print(f"Resuming from {npy_path}")
    else:
        matrix = np.full((n_layers, n_cats), np.nan)

    def get_pairs(cat: str) -> list:
        pairs = semantic_prompts.get(cat, [])
        return pairs[:args.n_pairs]

    block_list = (
        [("mm", i)     for i in range(N_MM)] +
        [("single", i) for i in range(N_SINGLE)]
    )

    for global_idx, (block_type, layer_idx) in enumerate(block_list):
        if not np.any(np.isnan(matrix[global_idx])):
            print(f"  [{global_idx:2d}] {block_type}-{layer_idx}: cached")
            continue

        tag = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
        print(f"  [{global_idx:2d}] {tag} ...", flush=True)

        for cat_idx, cat in enumerate(CATEGORIES):
            if not np.isnan(matrix[global_idx, cat_idx]):
                continue
            pairs = get_pairs(cat)
            sensitivities = []
            for pair in pairs:
                s = semantic_sensitivity(
                    layer_idx, block_type, pair,
                    pipe, clip_model, clip_processor,
                    args.device, args.n_steps,
                )
                sensitivities.append(s)
            matrix[global_idx, cat_idx] = float(np.mean(sensitivities))

        print(f"     {dict(zip(CATEGORIES, matrix[global_idx].tolist()))}")
        np.save(npy_path, matrix)

    # Save JSON version
    result_dict = {
        "categories": CATEGORIES,
        "layers": [f"MM-{i}" for i in range(N_MM)] + [f"S-{i}" for i in range(N_SINGLE)],
        "matrix": matrix.tolist(),
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
    parser.add_argument("--n_pairs",    type=int, default=10,
                        help="How many contrastive pairs to use per category (max 50)")
    parser.add_argument("--n_steps",    type=int, default=15)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
