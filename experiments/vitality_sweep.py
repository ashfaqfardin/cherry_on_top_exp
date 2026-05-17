"""
Experiment 1 — Layer vitality sweep.

For each of the 57 FLUX blocks (19 MM-DiT + 38 single-stream), bypass the
block during generation and measure how much the output changes via DINOv2
cosine similarity.  Low similarity → high vitality.

Outputs
-------
results/vitality_scores.json
    {
        "mm":     {"0": [sim, sim, ...], "1": [...], ..., "18": [...]},
        "single": {"0": [...], ..., "37": [...]}
    }
    Each list has N_PROMPTS × N_SEEDS entries.

Usage
-----
python experiments/vitality_sweep.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    [--n_steps 15] [--cpu_offload] [--device cuda]
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_MM = 19
N_SINGLE = 38
SEEDS = [0, 1, 2, 3]

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
    """Load DINOv2 ViT-B/14 via torch.hub."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def dino_similarity(img_a: Image.Image, img_b: Image.Image, dino_model,
                    device: str = "cuda") -> float:
    """Return cosine similarity between DINOv2 CLS embeddings of two PIL images."""
    ta = DINO_PREPROCESS(img_a).unsqueeze(0).to(device)
    tb = DINO_PREPROCESS(img_b).unsqueeze(0).to(device)
    fa = dino_model(ta)
    fb = dino_model(tb)
    sim = F.cosine_similarity(fa, fb, dim=-1).item()
    return float(sim)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    output_path = "results/vitality_scores.json"

    # Resume from partial results if they exist
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming from {output_path}")
    else:
        results = {"mm": {}, "single": {}}

    # Load prompts
    with open("prompts/vitality_prompts.json") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    pipe = load_pipeline(args.model_path, args.hf_token, args.device,
                         args.cpu_offload)
    dino = load_dino(args.device)

    def score_layer(block_type: str, layer_idx: int):
        """Return list of DINOv2 cosim scores across all prompts × seeds."""
        key = str(layer_idx)
        scores = []
        for prompt in prompts:
            # Reference image (full model, fixed seed 0)
            ref_img = generate_with_bypass(
                pipe, prompt, seed=0,
                block_type=block_type, bypass_idx=None,
                num_inference_steps=args.n_steps, device=args.device
            )
            for seed in SEEDS:
                ablated_img = generate_with_bypass(
                    pipe, prompt, seed=seed,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device
                )
                sim = dino_similarity(ref_img, ablated_img, dino, args.device)
                scores.append(sim)
        return scores

    # --- Double-stream (MM-DiT) blocks ---
    for i in range(N_MM):
        key = str(i)
        if key in results["mm"]:
            print(f"  MM block {i:2d}: cached, skipping")
            continue
        print(f"  MM block {i:2d} / {N_MM-1} ...", end=" ", flush=True)
        scores = score_layer("mm", i)
        results["mm"][key] = scores
        mean = float(np.mean(scores))
        print(f"mean sim = {mean:.4f}")
        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # --- Single-stream blocks ---
    for i in range(N_SINGLE):
        key = str(i)
        if key in results["single"]:
            print(f"  Single block {i:2d}: cached, skipping")
            continue
        print(f"  Single block {i:2d} / {N_SINGLE-1} ...", end=" ", flush=True)
        scores = score_layer("single", i)
        results["single"][key] = scores
        mean = float(np.mean(scores))
        print(f"mean sim = {mean:.4f}")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX layer vitality sweep")
    parser.add_argument("--model_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=15,
                        help="Inference steps per image (default 15)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
