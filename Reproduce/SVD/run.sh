#!/usr/bin/env bash
# SVD Style Personalization on Infinity-8B
# Can be run from anywhere: bash Reproduce/SVD/run.sh

# Resolve repo root from the script's own location
cd "$(dirname "$0")/../.." || exit 1

# ── Clone Infinity repo if not already present ───────────────────────────────
if [ ! -d "Infinity" ]; then
    git clone https://github.com/FoundationVision/Infinity.git Infinity
fi

# ── Run all style personalization configs ────────────────────────────────────
python Reproduce/SVD/run_svd_style.py \
    --hf_token "$HF_TOKEN" \
    --infinity_repo Infinity \
    --model_size 8b \
    --config prompts/reproduce_svd_style.json \
    --device cuda \
    --cache_dir ./models --save_images
