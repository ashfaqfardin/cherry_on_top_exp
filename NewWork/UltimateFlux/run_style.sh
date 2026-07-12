#!/usr/bin/env bash
# Task 7 — Reference-based style personalization (SVD-Style PFB + SAC)
# Transfers the visual style of a reference image to a text-prompted generation.
# Place style reference images in inputs/ before running.
#
# PFB (Principal Feature Blending):
#   Applied at double-stream block 1 (peak style/texture/object sensitivity).
#   pfb_steps_frac=(0.0, 0.25) — first 25% of denoising steps only.
#   Applying at all steps over-injects style features → noisy output.
#   pfb_alpha=1.0: exponential SVD reweighting, keeps top principal directions.
#   Lower alpha (e.g. 0.5) retains more style detail; raise for coarser transfer.
#
# SAC (Structural Attention Correction):
#   Copies image-token Q, K from source branch to edit branch at block 1.
#   Prevents PFB from distorting spatial layout and object structure.
#
# Style extraction: style image encoded as near-clean latent (t=0.1, 10% noise)
#   then one forward pass captures block 1 hidden states.  Low noise ensures
#   style features reflect the reference image content, not noise artifacts.
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
    --pfb_alpha 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name flower_watercolor \
    --prompt "A flower, watercolor painting" \
    --style_image inputs/watercolor_ref.png \
    --pfb_alpha 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name castle_oilpainting \
    --prompt "A castle, oil painting" \
    --style_image inputs/oilpainting_ref.jpg \
    --pfb_alpha 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name robot_3d \
    --prompt "A robot, 3D render" \
    --style_image inputs/cartoon3d_ref.png \
    --pfb_alpha 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name cat_crayon \
    --prompt "A cat, kid crayon drawing" \
    --style_image inputs/drawing_ref.png \
    --pfb_alpha 1.0 \
    --seed 42

echo "=== Style personalization complete. Results in results/ultimateflux/ ==="
