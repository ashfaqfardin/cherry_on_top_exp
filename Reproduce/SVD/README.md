# SVD Style Personalization — Training-Free Style Transfer on Infinity-2B

> Based on "A Training-Free Style-Personalization via SVD-Based Feature Decomposition" (CVPR 2025
> Implemented from scratch — no official code.

Given a **style reference image** and a **text prompt**, generates a 1024×1024 image that follows the prompt while adopting the visual style of the reference.

All commands are run from the **repo root** (`e:/Cherry_on_top/`).

---

## How it works

The method runs Infinity-2B in a **dual-stream** (batch size 2):

- **Stream 0 — content path**: unmodified autoregressive generation.
- **Stream 1 — generation path**: modified at scales s=3..12.

Two mechanisms are applied to the generation path:

| Mechanism | When | What it does |
|---|---|---|
| **PFB** (Principal Feature Blending) | Scale s=3 only | Injects the dominant SVD component of the style image's features into the generation path's accumulated codes |
| **SAC** (Structural Attention Correction) | Scales s=3..12 | Replaces generation-path Q and K in every self-attention with the content-path Q and K, preserving spatial structure |

---

## Prerequisites

1. Clone the [Infinity repo](https://github.com/FoundationVision/Infinity) and add it to `PYTHONPATH`, or place it at `<repo_root>/Infinity/`.
2. Checkpoints (`infinity_2b_reg.pth` and `infinity_vae_d32.pth`) are **auto-downloaded** from `FoundationVision/infinity` on first run. Pass your HuggingFace token via `--hf_token` or set `HF_TOKEN` in the environment to avoid rate limits.
3. The T5 text encoder (`google/flan-t5-xl`) is loaded from HuggingFace automatically.

---

## Single run

```bash
python Reproduce/SVD/run_svd_style.py \
    --style_image  inputs/watercolor_ref.jpg \
    --prompt       "a cat sitting on a windowsill in watercolor painting style" \
    --hf_token     YOUR_HF_TOKEN \
    --seed 0 \
    --device cuda \
    --save_images
```

Checkpoints are downloaded automatically on first run to `./models/`. You can also point to existing files:

```bash
python Reproduce/SVD/run_svd_style.py \
    --infinity_path  models/infinity_2b_reg.pth \
    --vae_path       models/infinity_vae_d32.pth \
    --style_image    inputs/watercolor_ref.jpg \
    --prompt         "a cat sitting on a windowsill in watercolor painting style" \
    --seed 0 --device cuda --save_images
```

Output: `results/svd_style/output/generated.png`

---

## Config file (multiple runs)

```bash
python Reproduce/SVD/run_svd_style.py \
    --config prompts/reproduce_svd_style.json \
    --device cuda \
    --save_images
```

Output per run: `results/svd_style/{name}/generated.png`

### Config format

```json
{
  "global": {
    "pfb_alpha": 1.0,
    "cfg": 3.0,
    "tau": 1.0,
    "seed": 0,
    "height": 1024,
    "width": 1024
  },
  "runs": [
    {
      "name": "cat_watercolor",
      "style_image": "inputs/watercolor_ref.jpg",
      "prompt": "a cat sitting on a windowsill in watercolor painting style"
    }
  ]
}
```

Per-run keys override `global`. `style_image` is relative to the repo root.

---

## All flags

| Flag | Default | Description |
|---|---|---|
| `--hf_token` | env `HF_TOKEN` | HuggingFace access token for downloads |
| `--infinity_path` | `models/infinity_2b_reg.pth` | Infinity-2B transformer checkpoint (auto-downloaded) |
| `--vae_path` | `models/infinity_vae_d32.pth` | BSQ-VAE d32 checkpoint |
| `--t5_path` | `google/flan-t5-xl` | HuggingFace ID or local path for T5 |
| `--cache_dir` | `./models` | Directory for model weight cache |
| `--style_image` | — | Path to reference style image |
| `--prompt` | — | Text prompt for the generated image |
| `--name` | `output` | Output subfolder name |
| `--pfb_alpha` | `1.0` | SVD reweighting factor α (paper default) |
| `--cfg` | `3.0` | Classifier-free guidance scale |
| `--tau` | `1.0` | Sampling temperature |
| `--top_k` | `900` | Top-k for token sampling |
| `--top_p` | `0.97` | Nucleus sampling threshold |
| `--cfg_insertion_layer` | `-5` | Layer index for CFG insertion (negative = from end) |
| `--seed` | `0` | Random seed |
| `--height` / `--width` | `1024` | Output resolution |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--out_dir` | `results/svd_style` | Root output directory |
| `--save_images` | off | Write output images to disk |
| `--config` | — | Path to JSON config file |
