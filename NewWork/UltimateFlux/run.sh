#!/usr/bin/env bash
# UltimateFlux — unified training-free editing on FLUX.1-dev
# All papers (FreeFlux, StableFlow, FluxSpace, SVD-Style) validated on FLUX.1-dev.
# Layer sets, guidance scale, and step count are paper-validated dev defaults.

set -euo pipefail

# Resolve repo root from this script's own location
cd "$(dirname "$0")/../.." || exit 1

# ── Dependencies ──────────────────────────────────────────────────────────────
# pip install -q diffusers transformers accelerate torchvision

# ── Common runner flags ───────────────────────────────────────────────────────
MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Non-rigid editing (FreeFlux content-similarity layers)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Task 1: Non-rigid editing ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name bird_perch_to_fly \
    --source_prompt "a bird perched on a branch" \
    --edit_prompt   "a bird flying away from the branch" \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task non_rigid \
    --name cat_sit_to_lie \
    --source_prompt "a cat sitting on a wooden floor" \
    --edit_prompt   "a cat lying down on a wooden floor" \
    --seed 42

# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Object addition (FreeFlux position-dependent layout layers)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Task 2: Object addition ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_add \
    --name table_add_vase \
    --source_prompt "a wooden dining table in a bright room" \
    --edit_prompt   "a wooden dining table with a vase of flowers in a bright room" \
    --added_word    "vase" \
    --seed 50

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_add \
    --name park_add_dog \
    --source_prompt "a park bench surrounded by trees" \
    --edit_prompt   "a park bench with a dog sitting next to it surrounded by trees" \
    --added_word    "dog" \
    --seed 55

# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Object replacement
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Task 3: Object replacement ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_replace \
    --name apple_to_orange \
    --source_prompt "a red apple on a wooden table" \
    --edit_prompt   "an orange on a wooden table" \
    --seed 10

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_replace \
    --name wooden_chair_to_metal \
    --source_prompt "a wooden chair in a bright room" \
    --edit_prompt   "a metal chair in a bright room" \
    --seed 15

# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Background replacement
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Task 4: Background replacement ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task bg_replace \
    --name cat_forest_to_beach \
    --source_prompt "a cat sitting in a forest" \
    --edit_prompt   "a cat sitting on a sunny beach" \
    --use_sam2 \
    --seed 20

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task bg_replace \
    --name dog_park_to_city \
    --source_prompt "a dog standing in a park" \
    --edit_prompt   "a dog standing on a city street" \
    --use_sam2 \
    --seed 25

# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Fine-grained attribute editing (FluxSpace orthogonal projection)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Task 5: Fine-grained attribute editing ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name man_add_glasses \
    --source_prompt "a portrait photo of a man" \
    --edit_prompt   "a portrait photo of a man wearing eyeglasses" \
    --seed 30

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name car_color_change \
    --source_prompt "a red sports car on a road" \
    --edit_prompt   "a blue sports car on a road" \
    --seed 40

# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — Reference-based style personalization (SVD-Style PFB + SAC)
# Requires style reference images in inputs/.
# ─────────────────────────────────────────────────────────────────────────────
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

echo "=== UltimateFlux complete. Results in results/ultimateflux/ ==="
