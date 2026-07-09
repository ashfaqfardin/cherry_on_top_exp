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
# Colour preservation at the attention level (SynPS + Untwisting RoPE inspired):
#
#   --v_blend 0.3  (default): At the 44 non-TIER_A (position-dependent) layers,
#   blend 30% source V into edit V: V_edit = 0.3*V_src + 0.7*V_edit_orig.
#   This is sub-cascade: the 70% edit contribution at every layer prevents the
#   residual from converging to source; the 30% provides direct colour grounding.
#   (100% V injection caused cascade; 0% left colour to drift — 30% is the balance.)
#
#   --k_s_lf 0.3  (default): Frequency-aware K blend at non-TIER_A layers.
#   After RoPE encoding, head dim d=0,1 are high-freq (most spatially sensitive);
#   d=D-2,D-1 are low-freq (most semantic/content-like). Source K weight interpolates
#   from 0% at d=0 (no spatial lock) to k_s_lf=30% at d=D-1 (semantic guidance).
#   Approximates Untwisting RoPE (arXiv:2602.05013) without de-rotation.
#
#   --preserve_color  (opt): Additionally apply Reinhard LAB statistics transfer
#   post-generation. Use only if v_blend alone is insufficient.
#
# Tuning:
#   --v_blend 0.5         More source colour at non-TIER_A (stronger but riskier).
#   --v_blend 0.1         Minimal colour injection (more pose freedom).
#   --k_s_lf 0.0          Disable K blend (V blend only, simpler).
#   --v_blend 0.0         Disable both (TIER_A K+V only; colour will drift).
#   --inject_steps_frac 0.08 1.0  Skip first 4 steps for drastic pose changes.
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
