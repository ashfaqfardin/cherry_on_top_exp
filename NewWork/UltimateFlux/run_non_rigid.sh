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
# Colour preservation — SynPS (CVPR 2026, arXiv:2512.14423):
#
#   Enabled by default (--synps; pass --no_synps or set "synps":false to disable).
#
#   At each TIER_A layer, source K is re-encoded with a w-scaled RoPE before injection:
#       cos_w = w * cos + (1 - w),  sin_w = w * sin
#   w=0: position-agnostic → edit Q attends by content (colour, texture), not location.
#   w=1: full RoPE → identical to FreeFlux baseline (spatial lock at TIER_A).
#
#   w is computed adaptively each denoising step from M_t = mean(S_img / S_txt):
#       M_t > m_max (0.95): under-editing → w=0  (colour retrieval mode)
#       M_t < m_min (0.70): over-editing  → w=1  (identity lock mode)
#       else:                w = (m_max - M_t) / (m_max - m_min)
#   First step starts at w=1.0 (no prior measurement).
#
#   WHY NO K BLEND AT NON-TIER_A: Source K at position-dependent layers corrupts
#   the Q×K attention pattern → attention shifts toward source spatial positions →
#   model generates source spatial structure → bird doesn't fly.
#
# Identity guidance — FFT frequency-selective latent blending:
#
#   Enabled with --identity_guidance.  After each denoising step in the first
#   identity_steps_frac of steps, the edit latent's low-frequency FFT components
#   are blended toward the source latent's:
#       fft_edit[:h_cut, :w_cut] = (1-s)*fft_edit + s*fft_src
#   Low-freq components (cutoff controlled by --low_freq_cutoff) encode global
#   colour and smooth gradients.  High-freq components (pose edges, fine texture)
#   are untouched.  This gives a complementary colour anchor at the latent level
#   that accumulates across steps — SynPS anchors at the attention level, this
#   anchors at the latent level.
#
#   Active in first 50% of steps only (--identity_steps_frac 0.0 0.5).  Later
#   steps refine pose-specific high-frequency detail free from the source anchor.
#
# Tuning:
#
#   Identity not preserved (colour of subject / background drifting):
#     Key insight: inject_steps_frac and v_blend_steps_frac are INDEPENDENT windows.
#     TIER_A K,V injection covers the main appearance anchor.
#     V-only blend at non-TIER_A fires even after inject window closes, catching
#     the final appearance-refinement steps that were previously completely free.
#
#     --inject_steps_frac 0.06 0.92   TIER_A K,V for 86% of steps.
#     --v_blend 0.25 --v_blend_steps_frac 0.0 1.0
#                                      Non-TIER_A V blend for ALL steps (including
#                                      the final 4 steps after TIER_A stops).
#     --identity_guidance --identity_strength 0.4 --identity_steps_frac 0.0 0.65
#                                      FFT low-freq latent anchor for first 65% of steps.
#     --static_w 0.0                   Max colour retrieval; bypass adaptive M_t.
#                                      Use for subtle poses where M_t stays high.
#
#   Pose not changing (still):
#     --inject_steps_frac 0.10 0.90   Skip more early steps; pose skeleton forms first.
#     --static_w 0.0                   Force position-agnostic; prevents TIER_A spatial lock.
#
#   Fallback (SynPS not helping / debugging):
#     --no_synps                        Disable SynPS; raw FreeFlux K+V at TIER_A.
#     --preserve_color                  Reinhard LAB post-processing (last resort).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=50
CFG=3.5
H=1024
W=1024

echo "=== Task 1: Non-rigid editing ==="

# Bird: drastic pose change (perched→flying).
#   inject_steps_frac 0.06 0.92: skip 3 steps for flying skeleton, then inject
#     TIER_A K,V for 46 steps — extended from 0.80 so appearance-refinement
#     steps are covered (was cutting off too early → colour drift in last 10 steps).
#   v_blend 0.25 all steps: V-only blend at non-TIER_A for ALL 50 steps.
#     Now decoupled from inject window (code fix) so it covers steps 46-49 where
#     TIER_A injection is off — these were the main source of colour drift.
#   identity_guidance: FFT low-freq latent anchor, first 65% of steps.
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name bird_perch_to_fly \
    --source_prompt "a bird perched on a branch" \
    --edit_prompt   "a bird flying away from the branch" \
    --inject_steps_frac 0.06 0.92 \
    --v_blend 0.25 --v_blend_steps_frac 0.0 1.0 \
    --identity_guidance --identity_strength 0.4 --identity_steps_frac 0.0 0.65 \
    --seed 42

# Cat: subtle pose change (sitting→lying).
#   static_w 0.0: bypass M_t adaptive schedule — for similar poses M_t stays
#     high, keeping SynPS w ≈ 1.0 (spatial lock). Forcing w=0 makes retrieval
#     purely content-based (colour/texture) rather than position-based.
#   inject_steps_frac 0.0 0.92: 3 extra free steps at end (was 0.70 — too short).
#   v_blend 0.25 all steps: covers steps 46-49 where TIER_A is off.
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name cat_sit_to_lie \
    --source_prompt "a cat sitting on a wooden floor" \
    --edit_prompt   "a cat lying down on a wooden floor" \
    --inject_steps_frac 0.0 0.92 \
    --static_w 0.0 \
    --v_blend 0.25 --v_blend_steps_frac 0.0 1.0 \
    --identity_guidance --identity_strength 0.4 --identity_steps_frac 0.0 0.65 \
    --seed 42

# Dog: moderate pose change (standing→jumping). Same strategy as cat.
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name dog_standing_to_jumping \
    --source_prompt "a golden retriever standing in a park" \
    --edit_prompt   "a golden retriever jumping in a park" \
    --inject_steps_frac 0.0 0.92 \
    --static_w 0.0 \
    --v_blend 0.25 --v_blend_steps_frac 0.0 1.0 \
    --identity_guidance --identity_strength 0.4 --identity_steps_frac 0.0 0.65 \
    --seed 7

echo "=== Non-rigid editing complete. Results in results/ultimateflux/ ==="
