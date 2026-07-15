"""
RealImageStable — Kontext mechanism on FLUX.1-schnell.

What Kontext does (from the paper):

  Visual stream receives two latent sequences:

    Context image latent  →  positional id  [T=1, h, w]
    Noisy output latent   →  positional id  [T=0, h, w]

  Both are concatenated into one token sequence and fed into the transformer.
  The 3D RoPE lets the model distinguish "what I'm generating" (T=0) from
  "what I'm conditioning on" (T=1) at every attention layer simultaneously.
  After denoising, only the T=0 (output) tokens are decoded.

Implementation here:

  1. Load FLUX.1-schnell (standard FluxPipeline, unmodified weights)
  2. VAE-encode input image  →  pack  →  context tokens  [T=1, h, w]
  3. Random noise            →  pack  →  output tokens   [T=0, h, w]
  4. Concatenate:  [context_tokens | output_tokens]  along sequence dim
  5. Concatenate:  [context_ids    | output_ids   ]  along sequence dim
  6. Run the schnell transformer denoising loop on the combined sequence
  7. After each step: update only output tokens, keep context tokens fixed
  8. Unpack and VAE-decode the output tokens  →  edited image

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

import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ──────────────────────────── helpers ─────────────────────────────────────────

def _preprocess(image: Image.Image, height: int, width: int) -> torch.Tensor:
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)


def _vae_encode(pipe: FluxPipeline, image: Image.Image,
                height: int, width: int, device: str) -> torch.Tensor:
    """PIL image → VAE latent  (1, 16, H//8, W//8)."""
    tensor = _preprocess(image, height, width).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        z = pipe.vae.encode(tensor).latent_dist.sample()
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return z


def _pack(z: torch.Tensor) -> torch.Tensor:
    """(1, 16, H, W) → (1, H//2 * W//2, 64)  — FLUX packing."""
    B, C, H, W = z.shape
    return (z.view(B, C, H // 2, 2, W // 2, 2)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, (H // 2) * (W // 2), C * 4))


def _unpack(z: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(1, seq, 64) → (1, 16, H, W)  — inverse of _pack."""
    H, W = height // 8, width // 8
    B, _, C4 = z.shape
    C = C4 // 4
    return (z.reshape(B, H // 2, W // 2, C, 2, 2)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(B, C, H, W))


def _make_image_ids(h_tokens: int, w_tokens: int, T: int, device) -> torch.Tensor:
    """
    3D positional ids for one latent block.

    Each token gets  (T, row, col)  where T distinguishes source:
      T=0  →  output  (noisy latent being denoised)
      T=1  →  context (clean input image, frozen)
    """
    ids = torch.zeros(h_tokens, w_tokens, 3)
    ids[..., 0] = T
    ids[..., 1] = torch.arange(h_tokens)[:, None]
    ids[..., 2] = torch.arange(w_tokens)[None, :]
    return ids.reshape(-1, 3).to(device)


def _vae_decode(pipe: FluxPipeline, z: torch.Tensor,
                height: int, width: int) -> Image.Image:
    z_dec = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        img = pipe.vae.decode(z_dec).sample
    img = ((img.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


# ──────────────────────────── pipeline ────────────────────────────────────────

def load_pipeline(model_path, device, hf_token=None, cache_dir=None):
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    ).to(device)
    return pipe


@torch.no_grad()
def run(pipe: FluxPipeline, args):
    input_image = Image.open(args.input).convert("RGB")
    device = pipe.device
    H, W   = args.height, args.width

    print(f"  input  : {args.input}  ({input_image.width}×{input_image.height})")
    print(f"  prompt : {args.prompt}")
    print(f"  seed={args.seed}  steps={args.num_steps}")

    # ── 1. Encode text ────────────────────────────────────────────────────────
    (prompt_embeds, pooled_embeds,
     text_ids) = pipe.encode_prompt(
        prompt=args.prompt,
        prompt_2=None,
        device=device,
        max_sequence_length=512,
    )

    # ── 2. Context tokens  [T=1, h, w] ───────────────────────────────────────
    ctx_z      = _vae_encode(pipe, input_image, H, W, str(device))
    ctx_packed = _pack(ctx_z)                               # (1, seq, 64)
    ctx_ids    = _make_image_ids(H // 16, W // 16, T=1, device=device)

    ctx_seq = ctx_packed.shape[1]

    # ── 3. Output tokens  [T=0, h, w] ────────────────────────────────────────
    g       = torch.Generator(device=device).manual_seed(args.seed)
    z       = randn_tensor((1, 16, H // 8, W // 8),
                           generator=g, device=device, dtype=torch.bfloat16)
    z_packed = _pack(z)                                     # (1, seq, 64)
    out_ids  = _make_image_ids(H // 16, W // 16, T=0, device=device)

    # ── 4. Set up scheduler with mu-shifted sigmas ───────────────────────────
    # FLUX.1-schnell requires a resolution-dependent shift (mu) applied to the
    # sigma schedule. Without it, the timestep distribution is wrong and the
    # output stays noisy. mu is computed from the OUTPUT sequence length only.
    out_seq_len = (H // 16) * (W // 16)
    mu = calculate_shift(
        out_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift",         0.5),
        pipe.scheduler.config.get("max_shift",          1.16),
    )
    sigmas = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps)
    pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    timesteps = pipe.scheduler.timesteps

    # ── 5. Denoising loop ─────────────────────────────────────────────────────
    for t in timesteps:
        # Concatenate context (T=1) + output (T=0) every step.
        # Context latent stays fixed; only z_packed is updated.
        combined        = torch.cat([ctx_packed, z_packed], dim=1)  # (1, 2*seq, 64)
        combined_ids    = torch.cat([ctx_ids,    out_ids  ], dim=0) # (2*seq, 3)

        timestep = t.expand(combined.shape[0]).to(combined.dtype) / 1000.0

        # guidance: dev uses 3.5 CFG embedding; schnell passes None
        guidance = (torch.full((combined.shape[0],), args.guidance_scale,
                               device=device, dtype=combined.dtype)
                    if args.guidance_scale > 1.0 else None)

        noise_pred = pipe.transformer(
            hidden_states         = combined,
            timestep              = timestep,
            encoder_hidden_states = prompt_embeds,
            pooled_projections    = pooled_embeds,
            txt_ids               = text_ids,
            img_ids               = combined_ids,
            guidance              = guidance,
            return_dict           = False,
        )[0]

        # Only the output (T=0) part is denoised — context stays frozen.
        noise_pred_out = noise_pred[:, ctx_seq:, :]

        z_packed = pipe.scheduler.step(
            noise_pred_out, t, z_packed, return_dict=False
        )[0]

    # ── 6. Decode output tokens ───────────────────────────────────────────────
    z_final = _unpack(z_packed, H, W)
    edited_image = _vae_decode(pipe, z_final, H, W)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
        in_path  = os.path.join(args.out_dir, "input.png")
        out_path = os.path.join(args.out_dir, "edited.png")
        input_image.resize((W, H), Image.LANCZOS).save(in_path)
        edited_image.save(out_path)
        print(f"  saved  → {in_path}")
        print(f"  saved  → {out_path}")

    return input_image, edited_image


# ──────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True)
    p.add_argument("--prompt",         required=True)
    p.add_argument("--num_steps",      type=int,   default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--height",         type=int,   default=1024)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--model_path",     default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--hf_token",       required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--cache_dir",      default="./models")
    p.add_argument("--out_dir",        default="results/realimageStable")
    p.add_argument("--save_images",    action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n[RealImageStable] Loading {args.model_path} …")
    pipe = load_pipeline(args.model_path, args.device, args.hf_token, args.cache_dir)
    print(f"[RealImageStable] Model loaded.\n")
    run(pipe, args)
    print("[RealImageStable] Done.")
