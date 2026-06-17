# Semantic Sensitivity Experiment

## Environment Setup

System libraries on HPC clusters can conflict with the pinned versions this project needs. Use an isolated virtual environment to avoid this.

**Python venv**

```bash
python -m venv cherry_env
source cherry_env/bin/activate      # run this every new session
```

> After activating, all `pip install` and `python` commands below will use the isolated environment and will not interfere with system packages.

---

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

> `--cpu_offload` offloads weights to CPU between inference steps — required on GPUs with limited VRAM (A100 40 GB or less). H100 can run without `--cpu_offload` for faster inference.
> Check disk space before running large models; clean the cache dir when full.

**`--n_pairs`** contrastive prompt pairs per semantic category (max 50).

**`--n_steps`** denoising steps per image.

**`--cache_dir`** directory for model weights and DINOv2 hub cache (default: `./models`).

**`--save_images`** save generated images to `results/images_{tag}/full/` (full-model) and `results/images_{tag}/MM-N/` (bypassed).

### FLUX.1-dev

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 28 \
    --device cuda --cpu_offload \
    --save_images \
    --cache_dir ./models
```

### FLUX.1-schnell

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-schnell \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 4 \
    --device cuda --cpu_offload \
    --save_images \
    --cache_dir ./models
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
    --save_images \
    --cache_dir ./models
```

## FluxSpace — Semantic Image Editing

> Based on [FluxSpace](https://github.com/gemlab-vt/FluxSpace) (Dalva et al., CVPR 2025).  
> Requires upstream diffusers — same as FLUX.2-dev:
> ```
> pip install -U diffusers
> ```

Edits a generated image by injecting an attribute into FLUX.1-dev's attention layers while preserving unrelated content.

**Parameter mapping (paper → CLI):**

| Paper symbol | CLI flag | Description |
|---|---|---|
| λ_coarse | `--edit_global_scale` | Global embedding shift (0–1) |
| λ_fine | `--edit_content_scale` | Per-block attention edit strength |
| τ_m | `--attention_threshold` | Cross-attention mask threshold (0–1) |
| start iter i | `--edit_start_iter` | First denoising step to apply edit |

**Global paper defaults:** `--n_steps 30 --seed 0 --attention_threshold 0.5`

### Reproduce all paper runs (config file)

```bash
python experiments/run_fluxspace.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_fluxspace.json \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

Saves one `original.png` + `edited.png` per run under `results/fluxspace/{name}/`.

---

### Eyeglasses (paper quantitative settings)

λ_coarse=0.8, λ_fine=5, start_iter=3

```bash
python experiments/run_fluxspace.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --prompt "portrait photo of a man" \
    --edit_prompt "eyeglasses" \
    --edit_global_scale 0.8 \
    --edit_content_scale 5 \
    --edit_start_iter 3 \
    --attention_threshold 0.5 \
    --n_steps 30 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

### Smile (paper quantitative settings)

λ_coarse=0.5, λ_fine=8, start_iter=5

```bash
python experiments/run_fluxspace.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --prompt "portrait photo of a man" \
    --edit_prompt "smiling" \
    --edit_global_scale 0.5 \
    --edit_content_scale 8 \
    --edit_start_iter 5 \
    --attention_threshold 0.5 \
    --n_steps 30 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

### Other paper edit prompts

Fine-grained: `sunglasses`, `beard`, `surprised`, `age`, `gender`, `overweight`, `clown makeup`  
Style: `comics style`, `3D cartoon style`, `anime style`, `cinematic lighting`  
Scene: `fall`, `snow`, `sunny`, `cherry blossom`, `raining`

For style edits the paper uses λ_coarse only (λ_fine=0, τ_m=0):
```bash
python experiments/run_fluxspace.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --prompt "portrait photo of a man" \
    --edit_prompt "comics style" \
    --edit_global_scale 0.5 \
    --edit_content_scale 0 \
    --attention_threshold 0 \
    --n_steps 30 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

Outputs saved to `results/fluxspace/seed{N}/original.png` and `edited.png`.

---

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
