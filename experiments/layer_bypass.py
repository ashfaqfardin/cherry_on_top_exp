"""
Layer bypass utilities — supports FLUX and SD3 model families.

Model          Pipeline class               Bypass mechanism
-----------    -------------------------    -----------------------------------------
FLUX.1-dev     FluxPipeline                 mm_skip_blocks / single_skip_blocks (built-in)
FLUX.1-schnell FluxPipeline                 same, guidance_scale=0.0
FLUX.2-dev     FluxPipeline                 same as FLUX.1-dev
SD 3.5 Large   StableDiffusion3Pipeline     monkey-patch block.forward (no built-in skip)

Usage
-----
from experiments.layer_bypass import load_pipeline, get_block_counts, generate_with_bypass

pipe = load_pipeline("stabilityai/stable-diffusion-3.5-large", hf_token, device)
n_mm, n_single = get_block_counts(pipe)   # (38, 0) for SD3.5

img = generate_with_bypass(pipe, prompt, seed=0, block_type="mm", bypass_idx=5)
"""

import contextlib
from typing import Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def detect_model_type(pipe) -> str:
    """Return 'flux' or 'sd3' based on pipeline class."""
    cls_name = type(pipe).__name__
    if "Flux" in cls_name:
        return "flux"
    if "StableDiffusion3" in cls_name:
        return "sd3"
    # Fallback: inspect transformer
    t = pipe.transformer
    if hasattr(t, "single_transformer_blocks"):
        return "flux"
    return "sd3"


def get_block_counts(pipe) -> tuple[int, int]:
    """
    Return (n_mm_blocks, n_single_blocks).

    FLUX.1-dev / schnell / FLUX.2-dev  → (19, 38)
    SD 3.5 Large                        → (38,  0)
    """
    model_type = detect_model_type(pipe)
    if model_type == "flux":
        n_mm     = len(pipe.transformer.transformer_blocks)
        n_single = len(pipe.transformer.single_transformer_blocks)
        return n_mm, n_single
    else:  # sd3
        n_mm = len(pipe.transformer.transformer_blocks)
        return n_mm, 0


def _default_guidance(pipe) -> float:
    """Return a sensible default guidance scale for the detected model."""
    model_type = detect_model_type(pipe)
    if model_type == "sd3":
        return 7.0
    # FLUX: schnell uses 0.0, dev/FLUX.2 use 3.5
    model_id = getattr(pipe, "name_or_path", "") or ""
    if "schnell" in model_id.lower():
        return 0.0
    return 3.5


# ---------------------------------------------------------------------------
# SD3 bypass context manager (monkey-patch block.forward)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _sd3_bypass_block(pipe, layer_idx: int):
    """
    Temporarily replace block.forward so it returns hidden_states and
    encoder_hidden_states unchanged (skips the block's transformation).

    SD3 JointTransformerBlock.forward() signature:
        (hidden_states, encoder_hidden_states, temb) → (hidden_states, encoder_hidden_states)
    """
    block = pipe.transformer.transformer_blocks[layer_idx]
    original_forward = block.forward

    def _skip(hidden_states, encoder_hidden_states, temb=None, **kwargs):
        return hidden_states, encoder_hidden_states

    block.forward = _skip
    try:
        yield
    finally:
        block.forward = original_forward


# ---------------------------------------------------------------------------
# Unified generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_bypass(
    pipe,
    prompt: str,
    seed: int = 0,
    *,
    block_type: str = "mm",       # "mm" or "single"
    bypass_idx: Optional[int] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: Optional[float] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Generate one image, optionally bypassing a single transformer block.

    Works for FLUX (dev, schnell, FLUX.2) and SD 3.5.

    Parameters
    ----------
    pipe          : loaded pipeline (FluxPipeline or StableDiffusion3Pipeline)
    prompt        : text prompt
    seed          : RNG seed
    block_type    : "mm" (double-stream / joint) or "single" (FLUX only)
    bypass_idx    : block index to bypass; None = full model
    guidance_scale: override auto-detected default
    """
    model_type = detect_model_type(pipe)
    gs = guidance_scale if guidance_scale is not None else _default_guidance(pipe)

    generator = torch.Generator(device=device).manual_seed(seed)

    if model_type == "flux":
        latents = torch.randn(
            (1, 4096, 64),
            generator=generator,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        mm_skip     = [bypass_idx] if (bypass_idx is not None and block_type == "mm")     else None
        single_skip = [bypass_idx] if (bypass_idx is not None and block_type == "single") else None

        result = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=gs,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            latents=latents,
            mm_skip_blocks=mm_skip,
            single_skip_blocks=single_skip,
        )
        return result.images[0]

    else:  # sd3
        # SD3 only has joint (MM-DiT) blocks — single-stream doesn't apply
        if bypass_idx is not None and block_type == "single":
            raise ValueError("SD 3.5 has no single-stream blocks. Use block_type='mm'.")

        ctx = (
            _sd3_bypass_block(pipe, bypass_idx)
            if bypass_idx is not None
            else contextlib.nullcontext()
        )
        with ctx:
            result = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=gs,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
        return result.images[0]


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(model_path: str, hf_token: str, device: str = "cuda",
                  cpu_offload: bool = False):
    """
    Auto-detect and load FluxPipeline or StableDiffusion3Pipeline.

    FLUX.2-dev workaround: the local diffusers fork predates AutoencoderKLFlux2.
    If that class is missing we pre-load the VAE as plain AutoencoderKL and inject
    it, bypassing the model-config class lookup.  mm_skip_blocks / single_skip_blocks
    still work because the pipeline itself comes from the local fork.
    """
    from diffusers import FluxPipeline, StableDiffusion3Pipeline, AutoencoderKL

    name = model_path.lower()

    if "stable-diffusion-3" in name or "sd3" in name:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            token=hf_token,
        )
    else:
        # FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev
        try:
            pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                visualize_attention=False,
                token=hf_token,
            )
        except AttributeError as exc:
            if "AutoencoderKLFlux2" not in str(exc) and "has no attribute" not in str(exc):
                raise
            print(
                f"[load_pipeline] Local diffusers fork missing {exc}.\n"
                "  Retrying with VAE pre-loaded as AutoencoderKL (FLUX.2-dev workaround).\n"
                "  Layer-bypass comparisons remain valid; absolute image quality may differ slightly."
            )
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.float16,
                token=hf_token,
            )
            pipe = FluxPipeline.from_pretrained(
                model_path,
                vae=vae,
                torch_dtype=torch.float16,
                visualize_attention=False,
                token=hf_token,
            )

    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    return pipe
