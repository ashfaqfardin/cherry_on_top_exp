"""
Shared model-loading helpers for all KontextEval phases.
"""

from __future__ import annotations

import torch
from diffusers import FluxKontextPipeline


def load_kontext_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
    hf_token: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "./models",
    cpu_offload: bool = False,
) -> FluxKontextPipeline:
    """Load FLUX.1-Kontext-dev pipeline and move to device."""
    pipe = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        token=hf_token,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(
    pipe: FluxKontextPipeline,
    prompt: str,
    image,
    seed: int = 42,
    num_steps: int = 28,
    guidance_scale: float = 2.5,
    height: int = 1024,
    width: int = 1024,
    prompt_2: str | None = None,
):
    """
    Single image edit call with a fixed seed.

    prompt   — fed to both CLIP (77-token limit) and T5. Keep ≤ 55 words.
    prompt_2 — fed to T5 only (no token limit). Use for extended detail:
               lighting, shadows, material descriptions, environment interaction.
               If None, T5 uses `prompt` as well.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        prompt_2=prompt_2,
        image=image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )
    return result.images[0]


def get_transformer(pipe: FluxKontextPipeline):
    """Return the underlying transformer model."""
    return pipe.transformer


def n_double_blocks(pipe: FluxKontextPipeline) -> int:
    return len(pipe.transformer.transformer_blocks)


def n_single_blocks(pipe: FluxKontextPipeline) -> int:
    return len(pipe.transformer.single_transformer_blocks)
