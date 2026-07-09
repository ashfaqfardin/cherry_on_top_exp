#!/usr/bin/env bash
# Task 1 — Non-rigid editing (FreeFlux mutual self-attention control)
#
# FreeFlux-validated settings (ICCV 2025):
#   TIER_A layers (13 content-similarity layers: 0,7,8,9,10,18,25,28,37,42,45,50,56):
#       Full K,V injection at ALL steps.
#   Steps: 50  — REQUIRED.  At 28 steps the free layers (44/57) do not accumulate
#       enough signal to shift the edit branch's pose against TIER_A anchoring.
#       50 steps gives 44 free-layer × 50 iterations vs 44 × 28, a 78% increase.
#
# Tuning:
#   --inject_steps_frac 0.15 1.0  Skip first ~8 steps (FreeFlux default start_step=4
#                                 out of 50). Helps if the edit is still too weak.
#   --inject_all_single          Add K-only injection at ALL single-stream layers.
#                                Use only if background drifts; K-only locks WHERE
#                                without locking V content, so pose can still emerge.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=50
CFG=3.5
H=1024
W=1024

echo "=== Task 1: Non-rigid editing ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name bird_perch_to_fly \
    --source_prompt "a bird perched on a branch" \
    --edit_prompt   "a bird flying away from the branch" \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name cat_sit_to_lie \
    --source_prompt "a cat sitting on a wooden floor" \
    --edit_prompt   "a cat lying down on a wooden floor" \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name dog_standing_to_jumping \
    --source_prompt "a golden retriever standing in a park" \
    --edit_prompt   "a golden retriever jumping in a park" \
    --seed 7

echo "=== Non-rigid editing complete. Results in results/ultimateflux/ ==="
