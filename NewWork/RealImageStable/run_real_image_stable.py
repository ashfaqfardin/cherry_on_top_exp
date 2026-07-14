"""
RealImageStable — Kontext-style real-image editing on FLUX.1-schnell.

How it works:
    FLUX.1 Kontext and FLUX.1-schnell share the same 12B DiT architecture.
    The Kontext mechanism is purely pipeline-level: the input image is
    VAE-encoded, packed, and its tokens are concatenated with the output
    latent tokens before the transformer sees them. A 3D RoPE offset
    (i=1 for context, i=0 for target) lets the model attend to both.

    Loading schnell weights into FluxKontextPipeline gives us:
      - Kontext's token concatenation mechanism (architecture is identical)
      - schnell's 4-step guidance-distilled sampler (fast inference)

    Context conditioning is best-effort (schnell wasn't fine-tuned on
    image pairs), but the model can still attend to context tokens through
    its self-attention, providing partial content guidance.

Pipeline:
    1. Load FLUX.1-schnell into FluxKontextPipeline
    2. Input image  →  VAE encode  →  pack  →  context tokens
    3. Random noise  →  pack  →  output tokens
    4. Concatenate: [context tokens | output tokens]  →  transformer
    5. Denoise with edit prompt (4 steps, guidance_scale=0.0)
    6. VAE-decode  →  edited image

Usage:
    python NewWork/RealImageStable/run_real_image_stable.py \\
        --hf_token "$HF_TOKEN" \\
        --input   inputs/cat.png \\
        --prompt  "make the cat's eyes blue" \\
        --seed 42 --save_images
"""

import argparse
import os
import sys

import torch
from PIL import Image
from diffusers import FluxKontextPipeline

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_pipeline(model_path, device, hf_token=None, cache_dir=None):
    # Load schnell weights into FluxKontextPipeline.
    # Both use the same 12B DiT — only the fine-tuning differs.
    pipe = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    ).to(device)
    pipe.set_progress_bar_config(desc="  denoising")
    return pipe


def run(pipe, args):
    input_image = Image.open(args.input).convert("RGB")
    print(f"  input  : {args.input}  ({input_image.width}×{input_image.height})")
    print(f"  prompt : {args.prompt}")
    print(f"  seed={args.seed}  steps={args.num_steps}")

    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

    # FluxKontextPipeline concatenates context image tokens with output tokens
    # internally — just pass the image directly.
    result = pipe(
        image               = input_image,
        prompt              = args.prompt,
        num_inference_steps = args.num_steps,
        guidance_scale      = 0.0,    # schnell is guidance-distilled
        height              = args.height,
        width               = args.width,
        generator           = generator,
        output_type         = "pil",
    )

    edited_image = result.images[0]

    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
        in_path  = os.path.join(args.out_dir, "input.png")
        out_path = os.path.join(args.out_dir, "edited.png")
        input_image.resize((args.width, args.height), Image.LANCZOS).save(in_path)
        edited_image.save(out_path)
        print(f"  saved  → {in_path}")
        print(f"  saved  → {out_path}")

    return input_image, edited_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True)
    p.add_argument("--prompt",         required=True)
    p.add_argument("--num_steps",      type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--height",         type=int,   default=1024)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--model_path",     default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--hf_token",       required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--cache_dir",      default="./models")
    p.add_argument("--out_dir",        default="results/realimageStable")
    p.add_argument("--save_images",    action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n[RealImageStable] Loading {args.model_path} into FluxKontextPipeline …")
    pipe = load_pipeline(args.model_path, args.device, args.hf_token, args.cache_dir)
    print(f"[RealImageStable] Model loaded.\n")
    run(pipe, args)
    print("[RealImageStable] Done.")
