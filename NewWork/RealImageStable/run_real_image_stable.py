"""
RealImageStable — StableFlow-style real-image editing on FLUX.1-schnell.

Pipeline:
    1. Load input image  (e.g. inputs/cat.png)
    2. VAE-encode  →  latent z_0
    3. Apply StableFlow nudge:  z_0 *= 1.15  (stabilises flow-matching inversion)
    4. Add flow-matching noise at the chosen strength:
           z_noisy = (1 - sigma) * z_0  +  sigma * eps
    5. Denoise with edit_prompt for the remaining steps (FLUX.1-schnell, 4 steps total)
    6. VAE-decode  →  edited image
    7. Save  input.png  and  edited.png

Usage:
    python NewWork/RealImageStable/run_real_image_stable.py \\
        --input   inputs/cat.png \\
        --prompt  "a cat with blue eyes" \\
        --strength 0.75 \\
        --seed 42 \\
        --save_images
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
from diffusers import FluxImg2ImgPipeline

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_NUDGE = 1.15   # StableFlow (arXiv:2502.XXXXX) latent scaling factor


# ─────────────────────────── helpers ──────────────────────────────────────────

def _preprocess(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Resize and normalise a PIL image to a [-1, 1] float tensor (1, 3, H, W)."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr   = np.array(image).astype(np.float32) / 255.0          # [0, 1]
    arr   = arr * 2.0 - 1.0                                      # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)


def _encode_and_nudge(pipe: FluxImg2ImgPipeline,
                      image: Image.Image,
                      height: int, width: int,
                      device: str) -> Image.Image:
    """
    VAE-encode the input image, apply the StableFlow nudge, then decode back
    to a PIL image so that FluxImg2ImgPipeline can re-encode it internally.

    Why decode back?
        FluxImg2ImgPipeline handles the noise-addition and sigma scheduling
        internally when given a PIL image. Passing a raw latent would require
        manually computing the sigma schedule — more error-prone.
        The round-trip (encode → nudge → decode) is perceptually lossless and
        lets the pipeline manage all scheduling correctly.
    """
    img_tensor = _preprocess(image, height, width).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        z = pipe.vae.encode(img_tensor).latent_dist.sample()
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z = z * _NUDGE                                           # StableFlow nudge
        # decode back: undo scaling and shift, then decode
        z_dec = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(z_dec).sample                  # (1, 3, H, W) in [-1, 1]

    decoded = ((decoded.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(decoded)


# ─────────────────────────── main ─────────────────────────────────────────────

def load_pipeline(model_path: str, device: str,
                  hf_token: str = None, cache_dir: str = None) -> FluxImg2ImgPipeline:
    pipe = FluxImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(desc="  denoising")
    return pipe


def run(pipe: FluxImg2ImgPipeline, args: argparse.Namespace):
    # ── Load input ──────────────────────────────────────────────────────────
    input_image = Image.open(args.input).convert("RGB")
    print(f"  input  : {args.input}  ({input_image.width}×{input_image.height})")
    print(f"  prompt : {args.prompt}")
    print(f"  strength={args.strength}  steps={args.num_steps}  seed={args.seed}")

    # ── VAE encode + StableFlow nudge ────────────────────────────────────────
    nudged_image = _encode_and_nudge(
        pipe, input_image, args.height, args.width, args.device
    )

    # ── Denoise with edit prompt ─────────────────────────────────────────────
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    result = pipe(
        prompt          = args.prompt,
        image           = nudged_image,
        strength        = args.strength,
        num_inference_steps = args.num_steps,
        guidance_scale  = 0.0,     # FLUX.1-schnell is guidance-distilled
        generator       = generator,
        height          = args.height,
        width           = args.width,
        output_type     = "pil",
    )

    edited_image = result.images[0]

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
        input_path  = os.path.join(args.out_dir, "input.png")
        edited_path = os.path.join(args.out_dir, "edited.png")

        input_image.resize((args.width, args.height), Image.LANCZOS).save(input_path)
        edited_image.save(edited_path)

        print(f"  saved  → {input_path}")
        print(f"  saved  → {edited_path}")

    return input_image, edited_image


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RealImageStable — StableFlow real-image editing on FLUX.1-schnell")

    p.add_argument("--input",      required=True,  help="Path to input image (e.g. inputs/cat.png)")
    p.add_argument("--prompt",     required=True,  help="Edit prompt (e.g. 'a cat with blue eyes')")
    p.add_argument("--strength",   type=float, default=0.75,
                   help="Noise strength: 0=no change, 1=ignore input image entirely (default 0.75)")
    p.add_argument("--num_steps",  type=int,   default=4,    help="Denoising steps (default 4 for FLUX.1-schnell)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--height",     type=int,   default=1024)
    p.add_argument("--width",      type=int,   default=1024)
    p.add_argument("--model_path", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--hf_token",   default=None)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--cache_dir",  default="./models")
    p.add_argument("--out_dir",    default="results/realimageStable")
    p.add_argument("--save_images", action="store_true", default=False)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n[RealImageStable] Loading {args.model_path} …")
    pipe = load_pipeline(args.model_path, args.device, args.hf_token, args.cache_dir)
    print(f"[RealImageStable] Model loaded on {args.device}\n")

    run(pipe, args)
    print("[RealImageStable] Done.")
