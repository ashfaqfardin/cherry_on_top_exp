#!/usr/bin/env bash
# Task 1 — Non-rigid editing: KV-Edit masked mode (ICCV 2025, arXiv:2502.17363)
#
# Mask derivation (priority order):
#
#   1. --fg_mask path/mask.png          User-supplied binary mask (highest quality).
#
#   2. --subject_noun "bird"            ConceptAttention (arXiv:2502.04320):
#                                       Uses FLUX's own attention output projections
#                                       in double-stream layers 8-17.  Similarity
#                                       between image-token and concept-word outputs
#                                       → token saliency map → threshold at mean.
#                                       No external model.  Finds the NAMED concept,
#                                       not just the largest/most-central blob.
#
#   3. --use_sam2_nonrigid              SAM2 automatic segmentation fallback.
#                                       Requires: pip install sam2
#
# Attention injection (TIER_A only; non-TIER_A completely free):
#
#   TIER_A layers (13 content-similarity, low-RoPE-freq):
#       Background:   K_edit[bg] ← K_src[bg],  V_edit[bg] ← V_src[bg]
#                     Content anchor only — no spatial lock.
#       Foreground:   K_edit[fg] ← SynPS(K_src_raw[fg], w)   appearance anchor
#                     V_edit[fg] ← V_src[fg]
#
#   Non-TIER_A layers (44 position-sensitive, high-RoPE-freq):
#       ALL tokens free — no injection.
#       Injecting source K here would RoPE-lock tokens to source positions,
#       preventing wings/limbs from extending into background space.
#
#   Background restoration: post_step soft compositing (dilated alpha blend).
#
# SynPS:
#   --static_w 0.0   fully position-agnostic K at TIER_A → strongest colour retrieval.
#   Adaptive w available via --m_min / --m_max (default: 0.7 / 0.95).
#
# Post-step soft composite:
#   --bg_dilate 6    dilate fg mask 6 tokens (~96px) before alpha-blending source bg
#                    latent into edit latent.  Prevents ghost traces from extended
#                    limbs being overwritten.  Increase if traces remain.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=50
CFG=3.5
H=1024
W=1024

echo "=== Task 1: Non-rigid editing ==="

# ── Case 1: bird perched → flying ────────────────────────────────────────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name bird_perch_to_fly \
    --source_prompt "a bird perched on a branch" \
    --edit_prompt   "a bird flying away from the branch" \
    --subject_noun  "bird" \
    --static_w 0.0 \
    --bg_dilate 6 \
    --seed 42

# ── Case 2: cat sitting → lying ──────────────────────────────────────────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name cat_sit_to_lie \
    --source_prompt "a cat sitting on a wooden floor" \
    --edit_prompt   "a cat lying down on a wooden floor" \
    --subject_noun  "cat" \
    --static_w 0.0 \
    --bg_dilate 6 \
    --seed 42

# ── Case 3: dog standing → jumping ───────────────────────────────────────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name dog_standing_to_jumping \
    --source_prompt "a golden retriever standing in a park" \
    --edit_prompt   "a golden retriever jumping in a park" \
    --subject_noun  "retriever" \
    --static_w 0.0 \
    --bg_dilate 6 \
    --seed 7

echo "=== Non-rigid editing complete. Results in results/ultimateflux/ ==="
