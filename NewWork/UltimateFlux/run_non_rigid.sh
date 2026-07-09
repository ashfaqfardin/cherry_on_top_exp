#!/usr/bin/env bash
# Task 1 — Non-rigid editing (two-tier K,V injection)
#
# Injection strategy:
#   Tier 1 — TIER_A (FreeFlux content-similarity layers, both stream types):
#       Full K,V injection at all steps → subject identity and appearance.
#   Tier 2 — ALL single-stream layers (19-56):
#       Full K,V injection at all steps → background scene locked to source.
#       Single-stream blocks refine texture/detail; injecting here prevents
#       background drift without affecting pose semantics.
#   Free double-stream layers [1-6, 11-17] (13 blocks):
#       Edit prompt drives new pose/action through text-image interaction here.
#
# Tuning:
#   --bg_steps_frac 0.5 1.0   Default: background lock starts at step 14/28.
#                              Raise START (e.g. 0.6 1.0) if pose change is weak.
#                              Lower START (e.g. 0.3 1.0) if background still drifts.
#   --no_inject_all_single    Use TIER_A only (FreeFlux original); may lose background.
#   --inject_steps_frac 0.0 0.8  Shorten TIER_A window if identity is over-locked.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
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
