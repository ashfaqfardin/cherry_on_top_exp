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
# Colour preservation at the attention level (SynPS CVPR 2026, arXiv:2512.14423):
#
#   --v_blend 0.3  (default): At the 44 non-TIER_A (position-dependent) layers,
#   blend 30% source V into edit V: V_edit = 0.3*V_src + 0.7*V_edit_orig.
#   V blend does NOT touch K — the Q×K attention pattern stays 100% edit-conditioned,
#   so the pose change (Q_edit attending differently from Q_src) is preserved.
#   The 30% V_src provides direct colour grounding at every non-TIER_A layer.
#
#   WHY NO K BLEND AT NON-TIER_A: Source K at position-dependent layers corrupts
#   the Q×K attention pattern → attention shifts toward source spatial positions →
#   model generates source spatial structure → bird doesn't fly.  Only V is blended.
#
# Tuning (if pose is suppressed by v_blend):
#   --v_blend_steps_frac 0.0 0.3   Limit V blend to first 15/50 steps only.
#                                   Colour is established early; pose develops freely later.
#   --v_blend 0.1                  Reduce V blend weight (less colour, more pose freedom).
#   --v_blend 0.0                  Disable V blend entirely (colour will drift).
#   --preserve_color               Add Reinhard LAB post-processing as fallback.
#   --inject_steps_frac 0.08 1.0   Skip first 4 TIER_A steps for drastic pose changes.
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
