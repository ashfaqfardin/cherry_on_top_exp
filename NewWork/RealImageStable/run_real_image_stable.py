"""
RealImageStable — Real-image editing on FLUX.1-schnell.

Pipeline:
    1. VAE-encode input image  →  z   (1, 16, H//8, W//8)
    2. Sample random noise     →  z1  (same shape)
    3. Start latent            =  z + z1
    4. Pack and denoise with edit_prompt (FLUX.1-schnell)
    5. VAE-decode  →  edited image

Usage:
    python NewWork/RealImageStable/run_real_image_stable.py \\
        --input   inputs/cat.png \\
        --prompt  "a cat with blue eyes" \\
        --seed 42 --save_images
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_pipeline(model_path, device, hf_token=None, cache_dir=None):
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    ).to(device)
    pipe.set_progress_bar_config(desc="  denoising")
    return pipe


def encode_image(pipe, image: Image.Image, height: int, width: int, device: str):
    """VAE-encode a PIL image → latent z  (1, 16, H//8, W//8)."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr   = np.array(image).astype(np.float32) / 255.0 * 2.0 - 1.0   # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        z = pipe.vae.encode(tensor).latent_dist.sample()
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return z   # (1, 16, H//8, W//8)


def pack(z: torch.Tensor) -> torch.Tensor:
    """Pack FLUX latent: (1, 16, H, W) → (1, H//2 * W//2, 64)."""
    B, C, H, W = z.shape
    return (z.view(B, C, H // 2, 2, W // 2, 2)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, (H // 2) * (W // 2), C * 4))


def run(pipe, args):
    input_image = Image.open(args.input).convert("RGB")
    print(f"  input  : {args.input}  ({input_image.width}×{input_image.height})")
    print(f"  prompt : {args.prompt}")
    print(f"  seed={args.seed}  steps={args.num_steps}")

    device = pipe.device

    # ── Step 1: VAE encode → z ───────────────────────────────────────────────
    z  = encode_image(pipe, input_image, args.height, args.width, str(device))

    # ── Step 2: Random noise → z1 ────────────────────────────────────────────
    g  = torch.Generator(device=device).manual_seed(args.seed)
    z1 = randn_tensor(z.shape, generator=g, device=device, dtype=z.dtype)

    # ── Step 3: Starting latent = z + z1, packed ────────────────────────────
    latents = pack(z + z1)   # (1, seq, 64)

    # ── Step 4: Denoise ──────────────────────────────────────────────────────
    result = pipe(
        prompt              = args.prompt,
        latents             = latents,
        num_inference_steps = args.num_steps,
        guidance_scale      = 0.0,   # FLUX.1-schnell is guidance-distilled
        height              = args.height,
        width               = args.width,
        output_type         = "pil",
    )

    edited_image = result.images[0]

    # ── Save ─────────────────────────────────────────────────────────────────
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
    p.add_argument("--input",       required=True)
    p.add_argument("--prompt",      required=True)
    p.add_argument("--num_steps",   type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--height",      type=int,   default=1024)
    p.add_argument("--width",       type=int,   default=1024)
    p.add_argument("--model_path",  default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--hf_token",    required=True)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/realimageStable")
    p.add_argument("--save_images", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n[RealImageStable] Loading {args.model_path} …")
    pipe = load_pipeline(args.model_path, args.device, args.hf_token, args.cache_dir)
    print(f"[RealImageStable] Model loaded.\n")
    run(pipe, args)
    print("[RealImageStable] Done.")
