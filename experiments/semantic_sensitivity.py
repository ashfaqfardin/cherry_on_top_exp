"""
Experiment 2 — Semantic specialisation of layers.

Uses the same methodology as the paper's vitality sweep:
  - DINOv2 perceptual similarity (not CLIP)
  - Contrastive pairs share the same seed so the only variable is the prompt
  - Sensitivity = |sim_full_pair − sim_ablated_pair|

Intuition
---------
Paper vitality asks:   "does this layer change the image?"
                        compare  same_prompt_full  vs  same_prompt_ablated

We ask:                "does this layer distinguish semantic category X?"
                        compare  pair_sim_full      vs  pair_sim_ablated

If bypassing layer L makes "red apple" and "green apple" look MORE similar
to DINOv2 (sim rises), then L was responsible for encoding colour.

Speed note
----------
Full-model images pre-generated ONCE before the layer loop (same pattern as
the paper's vitality pre-caching approach).

Quick-run (~15-20 min cpu_offload):  --n_pairs 1 --n_steps 4 --only_mm
Paper-style (all layers, 10 pairs):  --n_pairs 10 --n_steps 28

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
from torchvision import transforms  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.layer_bypass import generate_with_bypass, load_pipeline, get_block_counts, detect_model_type

CATEGORIES = ["colour", "style", "material", "texture", "shape", "layout", "object"]

DINO_PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# DINOv2 helpers  (same as vitality_sweep.py — paper's chosen metric)
# ---------------------------------------------------------------------------

def load_dino(device: str = "cuda"):
    torch.hub.set_dir("./models/torch_hub")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def dino_similarity(img_a: Image.Image, img_b: Image.Image,
                    dino_model, device: str) -> float:
    """DINOv2 cosine similarity between two images. Higher = more similar."""
    ta = DINO_PREPROCESS(img_a).unsqueeze(0).to(device)
    tb = DINO_PREPROCESS(img_b).unsqueeze(0).to(device)
    fa = dino_model(ta)
    fb = dino_model(tb)
    return float(F.cosine_similarity(fa, fb, dim=-1).item())


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def model_tag(model_path: str) -> str:
    """Derive a short filesystem-safe tag from a model path."""
    name = model_path.split("/")[-1]          # e.g. "FLUX.1-dev"
    return name.replace(".", "").replace("-", "_").lower()  # "flux1_dev"


def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    tag       = args.out_tag or model_tag(args.model_path)
    npy_path  = f"results/semantic_sensitivity_{tag}.npy"
    json_path = f"results/semantic_sensitivity_{tag}.json"
    print(f"Output tag: {tag}")

    with open("prompts/semantic_prompts.json") as f:
        semantic_prompts = json.load(f)

    pipe = load_pipeline(args.model_path, args.hf_token, args.device, args.cpu_offload)
    N_MM, N_SINGLE = get_block_counts(pipe)
    model_type = detect_model_type(pipe)
    print(f"Model type : {model_type}  |  MM blocks: {N_MM}  Single blocks: {N_SINGLE}")

    # SD3 has no single-stream blocks — silently cap
    if model_type == "sd3" and N_SINGLE == 0:
        print("  SD3 has no single-stream blocks — sweeping all MM blocks only.")

    block_list = [("mm", i) for i in range(N_MM)]
    if N_SINGLE > 0:
        block_list += [("single", i) for i in range(N_SINGLE)]
    n_layers = len(block_list)

    # Load or initialise result matrix
    if os.path.exists(npy_path):
        matrix = np.load(npy_path)
        if matrix.shape[0] != n_layers:
            print(f"  Saved matrix has {matrix.shape[0]} rows, expected {n_layers}. Starting fresh.")
            matrix = np.full((n_layers, len(CATEGORIES)), np.nan)
        else:
            print(f"Resuming from {npy_path}")
    else:
        matrix = np.full((n_layers, len(CATEGORIES)), np.nan)

    dino = load_dino(args.device)

    # ------------------------------------------------------------------
    # Pre-generate ALL full-model images for every pair — done ONCE
    # Each pair (a, b) uses the same seed so the only difference is the prompt.
    # Seed per pair = pair index (mirrors paper's per-prompt seed strategy).
    # ------------------------------------------------------------------
    print("Pre-generating full-model images for all pairs (seed = pair index)...")
    full_cache: dict = {}   # (cat, p_idx, 'a'/'b') → PIL Image
    for cat in CATEGORIES:
        pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
        for p_idx, (prompt_a, prompt_b) in enumerate(pairs):
            seed = p_idx          # unique seed per pair, same for both sides
            for side, prompt in [("a", prompt_a), ("b", prompt_b)]:
                full_cache[(cat, p_idx, side)] = generate_with_bypass(
                    pipe, prompt, seed=seed,
                    block_type="mm", bypass_idx=None,
                    num_inference_steps=args.n_steps, device=args.device,
                )
    print(f"  Cached {len(full_cache)} full-model images.\n")

    # Pre-compute full-model DINOv2 similarity for each pair (once)
    sim_full_cache: dict = {}   # (cat, p_idx) → float
    for cat in CATEGORIES:
        pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
        for p_idx in range(len(pairs)):
            sim_full_cache[(cat, p_idx)] = dino_similarity(
                full_cache[(cat, p_idx, "a")],
                full_cache[(cat, p_idx, "b")],
                dino, args.device,
            )

    # ------------------------------------------------------------------
    # Layer sweep — generate ablated images only, compare with DINOv2
    #
    # sensitivity = |sim_full_pair - sim_ablated_pair|
    #
    # High sensitivity: bypassing this layer made the two images look MORE
    # similar → that layer was responsible for the semantic difference.
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
                seed = p_idx      # same seed used when generating full-model refs

                img_a_abl = generate_with_bypass(
                    pipe, prompt_a, seed=seed,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )
                img_b_abl = generate_with_bypass(
                    pipe, prompt_b, seed=seed,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )

                sim_abl  = dino_similarity(img_a_abl, img_b_abl, dino, args.device)
                sim_full = sim_full_cache[(cat, p_idx)]

                # Positive delta: images became MORE similar after bypass
                # → this layer was encoding the semantic difference
                deltas.append(abs(sim_full - sim_abl))

            matrix[global_idx, cat_idx] = float(np.mean(deltas)) if deltas else 0.0

        print(f"     {dict(zip(CATEGORIES, [round(float(x), 4) for x in matrix[global_idx]]))}")
        np.save(npy_path, matrix)

    # Save JSON
    layer_labels = (
        [f"MM-{i}" for i in range(N_MM)] +
        [f"S-{i}" for i in range(N_SINGLE)]
    )[:n_layers]
    with open(json_path, "w") as f:
        json.dump({
            "metric":     "DINOv2 cosine similarity (same as paper vitality metric)",
            "categories": CATEGORIES,
            "layers":     layer_labels[:n_layers],
            "matrix":     matrix.tolist(),
        }, f, indent=2)

    print(f"\nDone. Saved:\n  {npy_path}\n  {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",    type=str, required=True)
    parser.add_argument("--n_pairs",     type=int, default=1,
                        help="Contrastive pairs per category (default 1; max 50)")
    parser.add_argument("--n_steps",     type=int, default=4,
                        help="Denoising steps (default 4 quick; use 28 for paper-style)")
    parser.add_argument("--out_tag",     type=str, default=None,
                        help="Tag for output filenames e.g. 'flux1_dev'. "
                             "Auto-derived from model name if omitted.")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
