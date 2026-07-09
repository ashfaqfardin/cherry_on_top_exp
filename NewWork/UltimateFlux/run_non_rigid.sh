#!/usr/bin/env bash
# Task 1 — Non-rigid editing (FreeFlux mutual self-attention control)
#
# Injection strategy — TIER_A (13 content-similarity layers), all 50 steps:
#   Source image-token K,V are copied into the edit branch at TIER_A layers only.
#   Q is never touched; it comes from the edit branch ("bird flying") and drives
#   the pose change by attending to source K,V differently than source Q does.
#
#   WHY TIER_A AND NOT ALL 57 LAYERS:
#   TIER_A layers are content-similarity-dependent (low RoPE frequency).  In these
#   layers, attention is driven by feature similarity, not spatial position.  The
#   edit Q therefore has freedom to attend differently from source Q.
#
#   The other 44 layers are position-dependent (high RoPE).  Injecting source K
#   there causes each edit token to attend mostly to its spatially-identical source
#   token (RoPE relative-position property), producing output pixel-identical to
#   source regardless of the edit prompt.  This is why ALL-57 fails.
#
#   Reference: FreeFlux run_non_rigid.py  layer_idx=[0,7,8,9,10,18,25,28,37,42,45,50,56]
#
# Colour preservation (preserve_v_all=True, default):
#   Source image-token V is also injected at the 44 non-TIER_A layers.
#   K is NOT injected there — only V.  This preserves the subject's colour/texture
#   without causing spatial locking (injecting K at those layers would).
#   Disable with --no_preserve_v if you want the model to generate a new colour.
#
# Tuning:
#   --inject_steps_frac 0.08 1.0  Skip first 4 steps (gives Q more divergence time).
#   --inject_steps_frac 0.0  0.8  Shorten window if background drifts too much.
#   --no_preserve_v               Let the model choose its own colour.
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
