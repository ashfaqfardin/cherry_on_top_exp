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

## Run Experiments

> Model weights are cached in `./models/`.
> `--cpu_offload` offloads weights to CPU between inference steps — required on GPUs with limited VRAM (A100 40 GB or less). H100 can run without `--cpu_offload` for faster inference.
> Check disk space before running large models; clean `./models/` when full.

**`--n_pairs`** contrastive prompt pairs per semantic category (max 50).

**`--n_steps`** denoising steps per image.

**`--save_images`** save generated images to `results/images_{tag}/full/` (full-model) and `results/images_{tag}/MM-N/` (bypassed).

### FLUX.1-dev

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 28 \
    --device cuda --cpu_offload \
    --save_images
```

### FLUX.1-schnell

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-schnell \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 4 \
    --device cuda --cpu_offload \
    --save_images
```

### FLUX.2-dev

> Requires `transformers>=4.52` and `diffusers` (latest). Restart runtime after upgrading.
```
!pip install -U diffusers
!pip install -U transformers
```

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.2-dev \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 28 \
    --device cuda --cpu_offload \
    --save_images
```

## Plot Results

```bash
python experiments/plot_semantic_heatmap.py --tag flux1_dev --threshold 0.92
python experiments/plot_semantic_heatmap.py --tag flux1_schnell --threshold 0.92
python experiments/plot_semantic_heatmap.py --tag flux2_dev --threshold 0.92
```

## Results

Zip the results folder (includes `.npy`, `.json`, plots, and saved images):

```bash
zip -r results.zip results/
```
