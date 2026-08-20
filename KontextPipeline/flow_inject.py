"""
flow_inject.py — Flow-Guided Trajectory Injection for FLUX Kontext.

Core mechanism (replaces K/V injection and 3-pass feature delta):

  At each denoising step t:
    v_scene = transformer(latents_t, reference=scene)
    v_obj   = transformer(latents_t, reference=obj_collage)
    v_final = v_scene * (1 - mask) + v_obj * mask
    latents_{t-1} = scheduler.step(v_final, t, latents_t)

  Background tokens follow the scene ODE trajectory (no change).
  Foreground tokens (inside SAM2 mask) follow the collage ODE trajectory (object inserted).
  The mask is the SAM2 segmentation of where the object will be placed.

This costs 2× the transformer FLOPs per step but requires zero capture passes,
zero K/V caches, and zero target_zone heuristics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── Latent packing helpers ────────────────────────────────────────────────────
# FLUX packs 2×2 patches of latents into a single token.
# (B, C, H, W) ↔ (B, H//2 × W//2, C×4)

def _pack(latents: torch.Tensor, patch: int = 2) -> torch.Tensor:
    B, C, H, W = latents.shape
    x = latents.view(B, C, H // patch, patch, W // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5)            # (B, H//2, W//2, C, 2, 2)
    return x.reshape(B, (H // patch) * (W // patch), C * patch * patch)


def _unpack(tokens: torch.Tensor, height: int, width: int, vae_sf: int, patch: int = 2) -> torch.Tensor:
    B, _, packed_C = tokens.shape
    C     = packed_C // (patch * patch)
    h_lat = height // vae_sf
    w_lat = width  // vae_sf
    h_tok = h_lat  // patch
    w_tok = w_lat  // patch
    x = tokens.view(B, h_tok, w_tok, C, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5)            # (B, C, h_tok, 2, w_tok, 2)
    return x.reshape(B, C, h_lat, w_lat)


def _image_ids(h_lat: int, w_lat: int, device, dtype, patch: int = 2) -> torch.Tensor:
    """
    Position IDs for packed image tokens. Shape (n_tok, 3).
    Format: [batch=0, row, col] where row/col are in the token grid.
    In Kontext, both gen and ref use the same grid (same spatial layout).
    """
    h_tok = h_lat // patch
    w_tok = w_lat // patch
    rows  = torch.arange(h_tok, device=device, dtype=dtype)
    cols  = torch.arange(w_tok, device=device, dtype=dtype)
    gr, gc = torch.meshgrid(rows, cols, indexing="ij")
    zeros  = torch.zeros_like(gr)
    return torch.stack([zeros, gr, gc], dim=-1).reshape(-1, 3)


# ── Image encoding ────────────────────────────────────────────────────────────

def _encode_image(pipe, img: Image.Image, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    """VAE-encode an image → latent tensor (1, C, h_lat, w_lat)."""
    device = pipe.device
    t = pipe.image_processor.preprocess(img.convert("RGB"), height=height, width=width)
    t = t.to(device=device, dtype=dtype)
    with torch.no_grad():
        z = pipe.vae.encode(t).latent_dist.sample()
    return (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


# ── Scheduler mu helper ───────────────────────────────────────────────────────

def _compute_scheduler_mu(pipe, n_tok: int):
    """
    Compute the dynamic-shifting mu for FlowMatchEulerDiscreteScheduler.
    FLUX Kontext uses use_dynamic_shifting=True, which requires mu at set_timesteps.
    mu is a linear interpolation of token count between base and max sequence lengths.
    Returns None if the scheduler doesn't need it (safe to call set_timesteps without mu).
    """
    sched = pipe.scheduler
    if not getattr(sched.config, "use_dynamic_shifting", False):
        return None

    # Try diffusers' own helper first
    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
        return calculate_shift(
            n_tok,
            sched.config.base_image_seq_len,
            sched.config.max_image_seq_len,
            sched.config.base_shift,
            sched.config.max_shift,
        )
    except (ImportError, AttributeError):
        pass

    # Fallback: replicate the linear formula directly
    base_seq   = getattr(sched.config, "base_image_seq_len", 256)
    max_seq    = getattr(sched.config, "max_image_seq_len",  4096)
    base_shift = getattr(sched.config, "base_shift",         0.5)
    max_shift  = getattr(sched.config, "max_shift",          1.16)
    m  = (max_shift - base_shift) / (max_seq - base_seq)
    b  = base_shift - m * base_seq
    return n_tok * m + b


# ── Dual-pass flow-guided injection ──────────────────────────────────────────

@torch.no_grad()
def run_flow_guided_injection(
    pipe,
    scene:        Image.Image,        # current scene (background reference)
    prompt:       str,
    obj_collage:  Image.Image,        # scene with pasted object (object reference)
    sam2_mask:    np.ndarray,         # (H, W) uint8 or float, 1 = object zone
    seed:         int,
    num_steps:    int,
    guidance:     float,
    height:       int,
    width:        int,
) -> Image.Image:
    """
    Dual-pass ODE step with mask-guided velocity blending.

    v_scene and v_obj are both computed at every denoising step.
    The final velocity is spatially blended using the SAM2 mask so that:
      - background tokens follow the scene's denoising trajectory
      - edit-zone tokens follow the collage's denoising trajectory
    """
    device = pipe.device
    dtype  = next(pipe.transformer.parameters()).dtype
    vae_sf = pipe.vae_scale_factor
    h_lat  = height // vae_sf
    w_lat  = width  // vae_sf
    h_tok  = h_lat  // 2           # FLUX 2×2 pack factor
    w_tok  = w_lat  // 2
    n_tok  = h_tok * w_tok         # gen tokens

    print(f"  [FLOW] Setup: {height}×{width}  "
          f"latents {h_lat}×{w_lat}  tokens {h_tok}×{w_tok}={n_tok}")

    # ── 1. Text conditioning ───────────────────────────────────────────────
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device,
        num_images_per_prompt=1, max_sequence_length=512,
    )

    # ── 2. Encode reference images ─────────────────────────────────────────
    scene_z   = _encode_image(pipe, scene,       height, width, dtype)
    collage_z = _encode_image(pipe, obj_collage, height, width, dtype)
    print(f"    [FLOW] References encoded. Latent shape: {tuple(scene_z.shape)}")

    # Pack references to token format
    scene_tok   = _pack(scene_z)        # (1, n_tok, 64)
    collage_tok = _pack(collage_z)

    # Position IDs: gen and ref use the same spatial grid in Kontext
    single_ids   = _image_ids(h_lat, w_lat, device, dtype)   # (n_tok, 3)
    combined_ids = torch.cat([single_ids, single_ids], dim=0)  # (2×n_tok, 3)

    # ── 3. Initial noisy latents ───────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(seed)
    # Use pipe's prepare_latents if available, else generate noise directly
    if hasattr(pipe, "prepare_latents"):
        try:
            latents = pipe.prepare_latents(
                1, pipe.transformer.config.in_channels // 4,
                height, width, dtype, device, generator
            )
        except TypeError:
            # Older diffusers signature may differ
            latents = torch.randn(
                (1, 16, h_lat, w_lat), device=device, dtype=dtype, generator=generator
            )
    else:
        latents = torch.randn(
            (1, 16, h_lat, w_lat), device=device, dtype=dtype, generator=generator
        )
    print(f"    [FLOW] Initial latents: {tuple(latents.shape)}")

    # ── 4. SAM2 mask in token space ────────────────────────────────────────
    mask_np  = cv2.resize(
        sam2_mask.astype(np.float32), (w_tok, h_tok),
        interpolation=cv2.INTER_NEAREST,
    )
    # (1, n_tok, 1) for broadcasting over the packed-channel dimension
    mask_tok = torch.from_numpy(mask_np.reshape(1, n_tok, 1)).to(device, dtype)
    print(f"    [FLOW] Mask coverage: {mask_np.mean()*100:.1f}% of tokens")

    # ── 5. Guidance tensor ─────────────────────────────────────────────────
    guidance_t = torch.tensor([guidance], device=device, dtype=dtype)

    # ── 6. Scheduler ──────────────────────────────────────────────────────
    # FlowMatchEulerDiscreteScheduler with use_dynamic_shifting=True requires mu.
    # mu is a linear function of token count (same formula diffusers uses internally).
    mu = _compute_scheduler_mu(pipe, n_tok)
    if mu is not None:
        pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=device)

    # ── 7. Dual-pass denoising loop ────────────────────────────────────────
    for i, t in enumerate(pipe.scheduler.timesteps):
        # Pack current gen latents → token format
        gen_tok = _pack(latents)                              # (1, n_tok, 64)

        # Build combined [gen | ref] input for both passes
        input_A = torch.cat([gen_tok, scene_tok],   dim=1)   # (1, 2n, 64)
        input_B = torch.cat([gen_tok, collage_tok], dim=1)

        # Normalize timestep to [0, 1] as expected by FLUX transformer
        t_norm = (t.float() / 1000.0).to(dtype).expand(1)

        # ─ Pass A: scene reference → v_scene ─
        out_A = pipe.transformer(
            hidden_states=input_A,
            timestep=t_norm,
            guidance=guidance_t,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_ids,
            return_dict=False,
        )[0]
        v_scene = out_A[:, :n_tok, :]                        # gen portion only

        # ─ Pass B: collage reference → v_obj ─
        out_B = pipe.transformer(
            hidden_states=input_B,
            timestep=t_norm,
            guidance=guidance_t,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_ids,
            return_dict=False,
        )[0]
        v_obj = out_B[:, :n_tok, :]                          # gen portion only

        # ─ Velocity blend ─
        # v_final = v_scene * (1 - mask) + v_obj * mask
        v_final_tok = v_scene * (1.0 - mask_tok) + v_obj * mask_tok

        # Unpack blended velocity back to latent space for scheduler
        v_final = _unpack(v_final_tok, height, width, vae_sf)

        # ─ Scheduler step ─
        latents = pipe.scheduler.step(v_final, t, latents, return_dict=False)[0]

        if (i + 1) % 5 == 0 or i == 0:
            print(f"    [FLOW] Step {i+1}/{num_steps}  t={t.item():.0f}")

    # ── 8. Decode ──────────────────────────────────────────────────────────
    latents_out = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image_t     = pipe.vae.decode(latents_out, return_dict=False)[0]
    return pipe.image_processor.postprocess(image_t, output_type="pil")[0]


# ── Removal pass ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_flow_removal(
    pipe,
    scene_masked: Image.Image,   # scene with removed object region filled (from collage.build_removal_reference)
    prompt: str,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
) -> Image.Image:
    """
    Standard single-pass generation with removal reference.
    The SAM2-masked, blur-filled scene gives FLUX a clean background hint.
    No velocity blending needed — just let FLUX reconstruct from context.
    """
    from .utils import run_standard
    return run_standard(
        pipe=pipe, canvas=scene_masked, prompt=prompt,
        seed=seed, num_steps=num_steps, guidance=guidance,
        height=height, width=width,
    )
