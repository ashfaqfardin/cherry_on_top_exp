#!/usr/bin/env bash
# Task 7 — Identity-preserving style transfer (StyleID + StableFlow latent nudging)
#
# FORMULA:
#   z_T = (1 − σ) · z_src + σ · ε         (flow-matching interpolation)
#         └──────── source image ────────┘
#   σ = content_strength (default 0.85)
#
#   Both branches start from z_T.
#   Source branch denoises z_T → reconstructs source (K tracks real source structure).
#   Edit branch denoises z_T with style K,V → styled reconstruction of source.
#
# ── Layer-stratified injection ──────────────────────────────────────────────────
#
#   Identity layers {0,7,8,9,10} — Attn(Q_edit, K_src, V_style):
#     K from SOURCE branch (real source denoising trajectory) → edit attends to
#     the same spatial regions as source reconstruction → identity preserved.
#     Style V reshapes appearance (colour, texture, brush strokes).
#
#   Texture layers {18,25,28,37,42,45,50,56} — Attn(Q_edit, K_style, V_style):
#     K+V from style reference → strong style in detail-refining single-stream layers.
#
# TIER_A safety: FreeFlux — all 13 layers are content-similarity-dependent
#   (low RoPE frequency), K injection never causes spatial locking.
#
# ── Key parameters ──────────────────────────────────────────────────────────────
#   --content_image     source image whose identity to preserve (required for img2img)
#   --style_image       reference image whose style to transfer
#   --content_strength  noise fraction (0.85 default): 0.6=strong identity, 0.95=strong style
#   --style_strength    K,V blend weight (1.0 default)
#
# Sources: StyleID arXiv:2312.09008, Z-STAR+ arXiv:2411.19231,
#          StableFlow §3 latent nudging, Scheduled Injection arXiv:2605.26538
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
