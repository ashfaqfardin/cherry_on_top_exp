#!/usr/bin/env bash
# Task 1 — Non-rigid editing (FreeFlux mutual self-attention control)
#
# Default injection strategy (FreeFlux-validated for FLUX.1-dev):
#   TIER_A layers (13 content-similarity layers, layers 0,7,8,9,10,18,25,28,37,42,45,50,56):
#       Full K,V injection at ALL 28 steps — preserves subject identity and background.
#       Low-RoPE-frequency layers encode WHAT without WHERE, so appearance is anchored
#       while the 13 free double-stream blocks [1-6, 11-17] let the edit prompt drive
#       the new pose through text-image interaction.
#
# Tuning:
#   --inject_all_single          Add K-only injection at ALL single-stream layers.
#                                Use if background still drifts after the default run.
#                                K-only locks attention positions (WHERE) without
#                                overriding content (V stays from edit branch), so
#                                pose can still emerge.
#   --inject_steps_frac 0.0 0.8  Shorten TIER_A window if the edit is too weak.
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
