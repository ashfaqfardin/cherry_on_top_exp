"""
Layer bypass utilities for FLUX vitality experiments.

The custom diffusers fork already supports mm_skip_blocks / single_skip_blocks
in the pipeline call. This module wraps that interface for convenience.

Key insight from transformer_flux.py:
- FluxTransformerBlock returns (encoder_hidden_states, hidden_states).
  When index_block is in mm_skip_blocks the block body is skipped entirely
  and both tensors pass through unchanged.
- FluxSingleTransformerBlock returns hidden_states.
  Same skip logic applies via single_skip_blocks.

Usage:
    from experiments.layer_bypass import generate_with_bypass

    img = generate_with_bypass(pipe, prompt, seed=0,
                               block_type="mm", bypass_idx=5)
"""

import torch
from PIL import Image


@torch.no_grad()
def generate_with_bypass(
    pipe,
    prompt: str,
    seed: int = 0,
    *,
    block_type: str = "mm",      # "mm" or "single"
    bypass_idx: int | None = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 15,
    guidance_scale: float = 3.5,
    device: str = "cuda",
) -> Image.Image:
    """
    Generate a single image, optionally bypassing one transformer block.

    Parameters
    ----------
    pipe         : FluxPipeline (already loaded and on device)
    prompt       : text prompt
    seed         : RNG seed for reproducibility
    block_type   : "mm"  → double-stream block (0-18)
                   "single" → single-stream block (0-37)
    bypass_idx   : block index to bypass; None = full model (no bypass)

    Returns
    -------
    PIL.Image
    """
    mm_skip = None
    single_skip = None

    if bypass_idx is not None:
        if block_type == "mm":
            mm_skip = [bypass_idx]
        elif block_type == "single":
            single_skip = [bypass_idx]
        else:
            raise ValueError(f"block_type must be 'mm' or 'single', got {block_type!r}")

    # Build deterministic latents matching run_stable_flow.py convention
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 4096, 64),
        generator=generator,
        device=device,
        dtype=pipe.transformer.dtype,
    )

    result = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        output_type="pil",
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        latents=latents,
        mm_skip_blocks=mm_skip,
        single_skip_blocks=single_skip,
    )
    return result.images[0]


def load_pipeline(model_path: str, hf_token: str, device: str = "cuda",
                  cpu_offload: bool = False):
    """Load FluxPipeline once and return it."""
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        visualize_attention=False,
        token=hf_token,
    )
    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)
    return pipe
