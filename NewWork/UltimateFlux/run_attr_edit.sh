#!/usr/bin/env bash
# Task 5 — Fine-grained attribute editing
#
# inject_layers controls the editing mode:
#   (default / omit)        → _PRESERVE_LAYERS (TIER_A + HOTSPOT, 20 layers)
#                              Tightest identity lock — for ADDING an attribute
#                              (glasses, hat, beard).
#   --inject_layers double_stream
#                           → Lock all 19 double-stream (joint text-image) blocks
#                              with K+V injection; 38 single-stream blocks are FREE.
#                              Recommended for COLOUR changes — text drives colour in
#                              single-stream with no mask, no overlay artifacts.
#                              --inject_steps_frac [0.0, 1.0] (default: all steps).
#                              If colour change is weak: try [0.0, 0.7].
#
#   --inject_layers color   → ColorCtrl (arXiv:2508.09131) — single-stream only.
#                              Double-stream blocks (0-18): standard SDPA, untouched.
#                              Single-stream blocks (19-56): manual attention with:
#                                Structure: source v-v pre-softmax scores → target.
#                                Colour: binary V mask (editing-region mask from
#                                  img→text attention; source V for non-editing).
#                              --top_k_frac 0.1–0.4 = editing region size.
#                              --color_word 'blonde'/'blue' = focus mask on that word.
#                              --qk_frac / --v_frac = step fraction for each component.
#                              --chunk_size = heads per manual-attn chunk (OOM → 2/1).
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

# ── Change colour (double_stream: lock 19 joint blocks, free 38 single blocks) ─
# inject_layers double_stream:
#   Locks all 19 double-stream (joint text-image) blocks with K+V injection.
#   All 38 single-stream blocks are completely free — the edit text ("blonde",
#   "blue") drives colour there without any mask or boundary effect.
#   Fixes two ColorCtrl artifacts at qk_frac=0:
#     - "adding new colour over old" (V-mask binary boundary blending)
#     - forehead/structure drift (unconstrained single-stream spatial layout)
#
# If colour change is weak: lower inject_steps_frac end (e.g. [0.0, 0.7]) so
# later steps are fully free. If structure drifts: raise end back toward 1.0.

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name woman_hair_color \
    --source_prompt "a woman with black hair" \
    --edit_prompt   "a woman with blonde hair" \
    --inject_layers double_stream \
    --seed 35

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name car_color_change \
    --source_prompt "a red sports car on a road" \
    --edit_prompt   "a blue sports car on a road" \
    --inject_layers double_stream \
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
