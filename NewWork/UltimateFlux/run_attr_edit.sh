#!/usr/bin/env bash
# Task 5 — Fine-grained attribute editing
#
# inject_layers controls the editing mode:
#   (default / omit)        → _PRESERVE_LAYERS (TIER_A + HOTSPOT, 20 layers)
#                              Tightest identity lock — for ADDING an attribute
#                              (glasses, hat, beard).
#   --inject_layers color   → ColorCtrl (arXiv:2508.09131) — single-stream only.
#                              Double-stream blocks (0-18): standard SDPA, untouched.
#                              Single-stream blocks (19-56): manual attention with:
#                                Structure: source v-v pre-softmax scores → target
#                                  (locks spatial layout without touching colour V).
#                                Colour: editing-region mask from target img→text
#                                  attention; source V copied to NON-editing tokens;
#                                  editing region keeps target V (new colour).
#                              --top_k_frac 0.1–0.4 = editing region size (20% default).
#                              --color_word 'blonde'/'blue' = focus mask on that word.
#                              --qk_frac / --v_frac = step fraction for each component.
#                              --chunk_size = heads per manual-attn chunk (OOM → 2 or 1).
#   --inject_layers tier_a  → TIER_A only (13 layers)
#                              Appearance preserved, position flexible —
#                              for shape-linked edits (breed change).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 5: Fine-grained attribute editing ==="

# ── Change colour (ColorCtrl: arXiv:2508.09131) ───────────────────────────────
# Tuning guide for color editing:
#   top_k_frac  — editing region size (fraction of 4096 image tokens).
#                 0.2 = 20% ≈ 820 tokens; too small → boundary hair/car stays old colour.
#                 0.35 is a safer default for hair; 0.4 for large subjects like cars.
#   qk_frac     — fraction of steps with v-v pre-softmax score injection (structure).
#                 1.0 = all steps → each editing token's output is ~80% source V (diluted).
#                 0.0 = disabled → V masking alone; colour changes fully, structure from noise.
#                 Try 0.0 first for strongest colour; raise if identity drifts too much.
#   mask_build_step — steps of free denoising before building the editing mask.
#                 5–8 = sufficient structure to locate the colour region.
#   v_frac      — steps with V masking active. 1.0 (all) is correct; lower only to debug.

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name woman_hair_color \
    --source_prompt "a woman with black hair" \
    --edit_prompt   "a woman with blonde hair" \
    --inject_layers color \
    --top_k_frac 0.35 \
    --qk_frac 0.0 \
    --color_word blonde \
    --mask_build_step 6 \
    --seed 35

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name car_color_change \
    --source_prompt "a red sports car on a road" \
    --edit_prompt   "a blue sports car on a road" \
    --inject_layers color \
    --top_k_frac 0.4 \
    --qk_frac 0.0 \
    --color_word blue \
    --mask_build_step 6 \
    --seed 40

# ── Add an accessory (strong identity preservation) ───────────────────────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name man_add_glasses \
    --source_prompt "a portrait photo of a man" \
    --edit_prompt   "a portrait photo of a man wearing eyeglasses" \
    --seed 30

# ── Shape-linked edit (tier_a — preserves appearance, loosens layout) ─────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name dog_breed_shape \
    --source_prompt "a labrador sitting on grass" \
    --edit_prompt   "a husky sitting on grass" \
    --inject_layers tier_a \
    --seed 45

echo "=== Attribute editing complete. Results in results/ultimateflux/ ==="
