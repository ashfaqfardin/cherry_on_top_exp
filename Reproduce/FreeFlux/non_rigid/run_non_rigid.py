"""
FreeFlux non-rigid image editing for FLUX.1-dev.

Based on: https://github.com/wtybest/FreeFlux (ICCV 2025)

The method performs training-free non-rigid editing (pose changes, deformations) by:
  1. Inverting the source image with DDIM-like forward inversion
  2. Denoising a batch of [source, target] with mutual self-attention control:
     image-token keys/values from the source are shared into the target at
     specified steps/layers, preserving structure while following the new prompt.

Usage — single run
------------------
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \\
    --hf_token "$HF_TOKEN" \\
    --source_image path/to/image.jpg \\
    --source_prompt "a cat sitting on a chair" \\
    --target_prompt "a cat jumping over a chair" \\
    --n_steps 28 --seed 0 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — config file
-------------------
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/reproduce_freeflux.json \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# Add repo root to sys.path so 'Reproduce' is importable as a package.
# Script lives at Reproduce/FreeFlux/non_rigid/run_non_rigid.py, so go up three levels.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Reproduce.FreeFlux.non_rigid.pipeline_flux_non_rigid import FluxPipeline
from Reproduce.FreeFlux.non_rigid.non_rigid_attn_utils import register_non_rigid_attention_control


def load_freeflux_pipeline(model_path: str, hf_token: str,
                            device: str = "cuda", cpu_offload: bool = False,
                            cache_dir: str = "./models",
                            start_step: int = 4, start_layer: int = 0,
                            n_steps: int = 28):
    """Load FluxPipeline and register the non-rigid attention processor."""
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )

    register_non_rigid_attention_control(
        pipe,
        start_step=start_step,
        start_layer=start_layer,
        total_steps=n_steps,
    )

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


def encode_image_to_latents(pipe, image_path: str, height: int, width: int,
                             device: str, dtype=torch.bfloat16):
    """VAE-encode a source image and return packed latents."""
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = 2.0 * image_tensor - 1.0  # [-1, 1]

    vae = pipe.vae
    execution_device = pipe._execution_device
    image_tensor = image_tensor.to(device=execution_device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        h_lat = height // pipe.vae_scale_factor
        w_lat = width // pipe.vae_scale_factor
        latents = FluxPipeline._pack_latents(
            latents, batch_size=1,
            num_channels_latents=latents.shape[1],
            height=h_lat, width=w_lat,
        )

    return latents


@torch.no_grad()
def run_non_rigid_edit(pipe, source_image_path: str,
                       source_prompt: str, target_prompt: str,
                       n_steps: int = 28, guidance_scale: float = 3.5,
                       height: int = 1024, width: int = 1024,
                       max_sequence_length: int = 512,
                       device: str = "cuda"):
    """Invert source image then generate the edited target."""
    dtype = torch.bfloat16

    # Encode source image
    src_latents = encode_image_to_latents(pipe, source_image_path, height, width, device, dtype)

    # Step 1: forward inversion with source prompt
    print("  Inverting source image...")
    inverted_latents = pipe(
        prompt=source_prompt,
        latents=src_latents.clone(),
        invert_image=True,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
    )

    # Step 2: edit — batch [src_prompt, tgt_prompt] with inverted latents for source
    print("  Generating edited image...")
    result = pipe(
        prompt=[source_prompt, target_prompt],
        latents=src_latents.repeat(2, 1, 1),
        inverted_latent_list=inverted_latents,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        output_type="pil",
    )

    # result.images[0] = source reconstruction, result.images[1] = edited
    return result.images[0], result.images[1]


def run_single(pipe, cfg: dict, out_dir: str, save_images: bool, device: str):
    """Run one experiment defined by cfg dict and optionally save images."""
    name            = cfg["name"]
    source_image    = cfg["source_image"]
    source_prompt   = cfg["source_prompt"]
    target_prompt   = cfg["target_prompt"]
    n_steps         = cfg.get("n_steps", 28)
    guidance_scale  = cfg.get("guidance_scale", 3.5)
    height          = cfg.get("height", 1024)
    width           = cfg.get("width", 1024)
    max_seq_len     = cfg.get("max_sequence_length", 512)

    print(f"\n[{name}]  source='{source_prompt}'  target='{target_prompt}'")
    print(f"         steps={n_steps}  guidance={guidance_scale}  size={height}x{width}")

    img_src_recon, img_edited = run_non_rigid_edit(
        pipe,
        source_image_path=source_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_seq_len,
        device=device,
    )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        img_src_recon.save(os.path.join(run_dir, "source_recon.png"))
        img_edited.save(os.path.join(run_dir, "edited.png"))
        # Also copy source image for reference
        src_img = Image.open(source_image).convert("RGB")
        src_img.save(os.path.join(run_dir, "source.png"))
        print(f"  Saved → {run_dir}/")

    return img_src_recon, img_edited


def load_config(config_path: str) -> list:
    """
    Load runs from a JSON config file.

    The JSON must have a "runs" list. An optional "global" dict provides
    default values that each run inherits unless it overrides them.
    """
    with open(config_path) as f:
        data = json.load(f)

    global_defaults = data.get("global", {})
    runs = []
    for run in data["runs"]:
        merged = {**global_defaults, **run}
        if "name" not in merged:
            raise ValueError(f"Each run must have a 'name' field: {run}")
        runs.append(merged)
    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="FreeFlux non-rigid image editing (ICCV 2025)"
    )

    # --- Config-file mode ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON run-config file. "
                             "When provided, --source_image/--source_prompt/--target_prompt are not required.")

    # --- Single-run mode ---
    parser.add_argument("--source_image",  type=str, default=None,
                        help="Path to the source image (required without --config)")
    parser.add_argument("--source_prompt", type=str, default=None,
                        help="Text description of the source image (required without --config)")
    parser.add_argument("--target_prompt", type=str, default=None,
                        help="Text description of the desired edit (required without --config)")
    parser.add_argument("--n_steps",             type=int,   default=28,
                        help="Denoising steps (default 28)")
    parser.add_argument("--guidance_scale",      type=float, default=3.5)
    parser.add_argument("--height",              type=int,   default=1024)
    parser.add_argument("--width",               type=int,   default=1024)
    parser.add_argument("--max_sequence_length", type=int,   default=512,
                        help="T5 max token length (default 512)")
    parser.add_argument("--start_step",  type=int, default=4,
                        help="First denoising step to apply attention sharing (default 4)")
    parser.add_argument("--start_layer", type=int, default=0,
                        help="First transformer layer to apply attention sharing (default 0)")

    # --- Infrastructure ---
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload weights to CPU between steps (saves VRAM)")
    parser.add_argument("--cache_dir",  type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="results/freeflux/non_rigid",
                        help="Output directory for saved images")
    parser.add_argument("--save_images", action="store_true",
                        help="Save source.png, source_recon.png, and edited.png for each run")

    args = parser.parse_args()

    # Validate: without --config, source args are required
    if args.config is None:
        if args.source_image is None or args.source_prompt is None or args.target_prompt is None:
            parser.error("--source_image, --source_prompt, and --target_prompt are required unless --config is provided.")

    return args


def main():
    args = parse_args()

    print("[FreeFlux] Loading FLUX.1-dev...")
    pipe = load_freeflux_pipeline(
        args.model_path, args.hf_token,
        device=args.device, cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
        start_step=args.start_step,
        start_layer=args.start_layer,
        n_steps=args.n_steps,
    )
    print("[FreeFlux] Model loaded. Non-rigid attention processor registered.")

    if args.config:
        runs = load_config(args.config)
        print(f"\n[FreeFlux] Running {len(runs)} experiment(s) from {args.config}")
        for cfg in runs:
            run_single(pipe, cfg,
                       out_dir=args.out_dir,
                       save_images=args.save_images,
                       device=args.device)
        print(f"\n[FreeFlux] All runs complete.")
        if args.save_images:
            print(f"  Results saved to {args.out_dir}/")
    else:
        cfg = {
            "name":               "output",
            "source_image":       args.source_image,
            "source_prompt":      args.source_prompt,
            "target_prompt":      args.target_prompt,
            "n_steps":            args.n_steps,
            "guidance_scale":     args.guidance_scale,
            "height":             args.height,
            "width":              args.width,
            "max_sequence_length": args.max_sequence_length,
        }
        run_single(pipe, cfg,
                   out_dir=args.out_dir,
                   save_images=args.save_images,
                   device=args.device)


if __name__ == "__main__":
    main()
