#!/usr/bin/env bash
# Task 7 — Reference-based style personalization (StyleID K,V injection)
# Transfers the visual style of a reference image to a text-prompted generation.
# Place style reference images in inputs/ before running.
#
# Mechanism (StyleID, CVPR 2024 arXiv:2312.09008 adapted to FLUX):
#
#   1. PRE-GENERATE: one FLUX forward pass on the style reference image
#      at t=0.5 (50/50 noise-clean mix, empty prompt) captures K,V at
#      all 13 TIER_A (content-similarity) attention layers.
#
#   2. GENERATION: layer-stratified injection at every denoising step:
#
#      Identity layers {0,7,8,9,10} — V-only:
#        Attn(Q_edit, K_edit, V_style)
#        Content Q×K pattern intact → character shape/pose preserved.
#        Style V reshapes appearance (colour, texture, brush strokes).
#
#      Texture layers {18,25,28,37,42,45,50,56} — K+V:
#        Attn(Q_edit, K_style, V_style)
#        Strong style in detail-refining layers; structure already committed.
#
# TIER_A safety: FreeFlux identifies these 13 layers as content-similarity-
#   dependent (low RoPE frequency) — K injection here never causes spatial
#   locking.  Non-TIER_A layers (high RoPE) are left entirely free.
#
# --style_strength 1.0  — K,V blend weight (0.0=no injection, 1.0=full style).
#   Reduce to 0.7–0.8 if character structure is still distorted.
#
# Why the old SVD/PFB approach was replaced:
#   PFB modified one block's hidden-state SVD at first 25% of steps.
#   SAC running after PFB (steps 25%–100%) forced edit Q,K → source Q,K
#   while edit hidden states were in a PFB-modified subspace — the mismatch
#   collapsed denoising to a plain constant color.  Direct K,V injection at
#   all TIER_A layers for all steps avoids this entirely.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 7: Style personalization ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name cat_watercolor \
    --prompt "A cat, watercolor painting" \
    --style_image inputs/watercolor_ref.png \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name flower_watercolor \
    --prompt "A flower, watercolor painting" \
    --style_image inputs/watercolor_ref.png \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name castle_oilpainting \
    --prompt "A castle, oil painting" \
    --style_image inputs/oilpainting_ref.jpg \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name robot_3d \
    --prompt "A robot, 3D render" \
    --style_image inputs/cartoon3d_ref.png \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name cat_crayon \
    --prompt "A cat, kid crayon drawing" \
    --style_image inputs/drawing_ref.png \
    --style_strength 1.0 \
    --seed 42

echo "=== Style personalization complete. Results in results/ultimateflux/ ==="
