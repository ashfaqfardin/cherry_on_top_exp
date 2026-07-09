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
# Colour preservation (preserve_color=True, default):
#   After generation, a Reinhard LAB statistics transfer is applied: the edit
#   image's per-channel LAB mean and std are matched to the source's.  This
#   corrects subject colour (bird/cat) without affecting the attention mechanism,
#   so there is no risk of the cascade collapse that V-injection at all 44 layers
#   would cause (V at all layers forces the residual toward source at every block,
#   cascading the hidden state to source regardless of the edit prompt).
#   Disable with --no_preserve_color if you want the model to generate a new colour.
#
# Tuning:
#   --inject_steps_frac 0.08 1.0  Skip first 4 steps (gives Q more divergence time).
#   --inject_steps_frac 0.0  0.8  Shorten window if background drifts too much.
#   --no_preserve_color           Let the model choose its own colour.
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
