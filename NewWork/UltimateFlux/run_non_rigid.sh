#!/usr/bin/env bash
# Task 1 — Non-rigid editing: KV-Edit masked mode (ICCV 2025, arXiv:2502.17363)
#
# Architecture: three-zone attention injection based on SAM2 subject mask.
#
#   Background tokens  (all 57 layers, all 50 steps):
#       K_edit[bg] ← K_src[bg],  V_edit[bg] ← V_src[bg]
#       Background is pixel-locked at the attention level.  Additionally,
#       after each denoising step, bg latent tokens are hard-overwritten with
#       the source latent (KV-Edit inversion-free compositing).
#
#   Foreground tokens at TIER_A  (13 content-similarity layers):
#       K_edit[fg] ← SynPS(K_src_raw[fg], w)   appearance anchor
#       V_edit[fg] ← V_src[fg]
#       Subject colour and texture are pulled from source at TIER_A where
#       attention is content-driven (not position-driven).
#
#   Foreground tokens at non-TIER_A  (44 position-dependent layers):
#       K_edit[fg], V_edit[fg] unchanged  (edit branch values)
#       Subject is completely free.  Edit Q from "bird flying" drives
#       the attention pattern without any source K competing — pose
#       changes naturally because no spatial lock is applied here.
#
# SAM2 mask: --use_sam2_nonrigid
#   Generates a source image preview (same seed), runs SAM2 automatic
#   segmentation to find the foreground subject, then uses the resulting
#   token mask to activate the three-zone injection above.
#   Requires: pip install sam2
#
# SynPS (CVPR 2026, arXiv:2512.14423):
#   At TIER_A, source K is re-encoded with w-scaled RoPE:
#       cos_w = w*cos + (1-w),   sin_w = w*sin
#   w=0: position-agnostic (colour by content similarity, max identity).
#   w=1: full RoPE (FreeFlux baseline, spatial lock at TIER_A).
#   Adaptive w from M_t = mean(S_img_fg / S_txt) over TIER_A blocks
#   (measured on foreground tokens only in masked mode).
#
# Tuning:
#   --use_sam2_nonrigid              Auto-segment subject; activates masked mode.
#   --fg_mask path/mask.png          Supply your own binary mask instead of SAM2.
#   --static_w 0.0                   Force position-agnostic K at every step.
#   --inject_steps_frac 0.0 1.0      Injection window for foreground TIER_A anchor.
#   --identity_guidance              FFT low-freq latent colour anchor (optional extra).
#   --no_synps                       Disable SynPS; raw source K at TIER_A fg tokens.
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
    --use_sam2_nonrigid \
    --static_w 0.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name cat_sit_to_lie \
    --source_prompt "a cat sitting on a wooden floor" \
    --edit_prompt   "a cat lying down on a wooden floor" \
    --use_sam2_nonrigid \
    --static_w 0.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name dog_standing_to_jumping \
    --source_prompt "a golden retriever standing in a park" \
    --edit_prompt   "a golden retriever jumping in a park" \
    --use_sam2_nonrigid \
    --static_w 0.0 \
    --seed 7

echo "=== Non-rigid editing complete. Results in results/ultimateflux/ ==="
