#!/usr/bin/env bash
# Task 5 — Fine-grained attribute editing
#
# inject_layers controls how strongly the source is preserved:
#   (default / omit)  → _PRESERVE_LAYERS (TIER_A + HOTSPOT, 20 layers)
#                        Strong identity lock — for ADDING an attribute
#                        (glasses, hat, beard).
#   --inject_layers hotspot → HOTSPOT only (7 layers)
#                        Spatial structure preserved, appearance FREE —
#                        for CHANGING colour or texture (hair, car, coat).
#   --inject_layers tier_a  → TIER_A only (13 layers)
#                        Appearance preserved, position flexible —
#                        for shape-linked edits (breed change).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 5: Fine-grained attribute editing ==="

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

# ── Change colour / texture (hotspot only — frees appearance) ─────────────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name woman_hair_color \
    --source_prompt "a woman with brown hair" \
    --edit_prompt   "a woman with blonde hair" \
    --inject_layers hotspot \
    --seed 35

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name car_color_change \
    --source_prompt "a red sports car on a road" \
    --edit_prompt   "a blue sports car on a road" \
    --inject_layers hotspot \
    --seed 40

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
