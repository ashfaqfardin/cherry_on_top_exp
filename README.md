# Semantic Sensitivity Experiment

## Setup

```bash
pip install transformers==4.44.2
```

```bash
cd cherry_on_top_exp/
pip install -e .
```

## HF Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

## Generate Prompts

Using built-in prompts (no API needed):

```bash
python generate_prompts.py
```

Using Qwen/Qwen3.5-27B via HuggingFace API (requires HF token):

```bash
python generate_prompts.py --qwen --hf_token "$HF_TOKEN"
```

## Run Experiments

> `--cpu_offload` applied for low RAM (50 GB system RAM is enough). H100 GPU may not require CPU offloading. Also check disk space, clean it when full.

**`--n_pairs`** contrastive prompt pairs per semantic category (max 50).

**`--n_steps`** denoising steps per image. 4 = fast, 28 = paper quality. FLUX.1-schnell is designed for 4 steps.

### FLUX.1-dev

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 28 \
    --device cuda --cpu_offload
```

### FLUX.1-schnell

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-schnell \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 4 \
    --device cuda --cpu_offload
```

## Plot Results

```bash
python experiments/plot_semantic_heatmap.py --tag flux1_dev --threshold 0.92
python experiments/plot_semantic_heatmap.py --tag flux1_schnell --threshold 0.92
```
