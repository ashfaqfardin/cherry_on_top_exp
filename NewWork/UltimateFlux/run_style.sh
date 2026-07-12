#!/usr/bin/env bash
# Task 7 — Identity-preserving style transfer (StyleID adapted for FLUX)
#
# FORMULA at ALL 13 TIER_A layers:
#   Attn(Q_edit, K_src, V_style)
#
#   K_src  — from source branch's own denoising trajectory
#             Edit attends to the same spatial regions as the unmodified source
#             → identity preserved (character, pose, shape) across all 13 layers
#
#   V_style — from style reference image (captured at t=0.3, 70% clean signal)
#             Attention returns style appearance values (colour, texture,
#             brushstrokes) across all 13 TIER_A layers
#
# Both problems solved simultaneously: identity locked via K_src everywhere,
# style transferred via V_style everywhere.
#
# Prior split (K_src+V_src at 5 identity layers, K_sty+V_sty at 8 texture
# layers) was insufficient: identity from only 5 layers, zero style at
# identity layers, K_sty at texture layers diverging edit from source.
#
# TIER_A safety: FreeFlux — all 13 layers are content-similarity-dependent
#   (low RoPE frequency), K injection never causes spatial locking.
#
# ── Key parameters ──────────────────────────────────────────────────────────────
#   --style_image       reference image whose style to transfer
#   --style_strength    V_style blend weight (1.0 = full style, default)
#   --content_image     (optional) source image for img2img mode
#   --content_strength  noise fraction (0.85 default): 0.6=strong identity, 0.95=strong style
#
# Sources: StyleID arXiv:2312.09008, FreeFlux (TIER_A layer safety)
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 7: Identity-preserving style transfer ==="
# Place your source image in inputs/source.png and style reference in inputs/*_ref.*
# Adjust --content_strength: lower (0.6) = stronger identity, higher (0.9) = stronger style.

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

echo "=== Style transfer complete. Results in results/ultimateflux/ ==="
