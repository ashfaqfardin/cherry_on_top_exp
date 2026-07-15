"""
RealImageStable — Training-free real image editing on FLUX.1-dev.

Option 1: Dual-branch attention injection with real image start.

Pipeline:
    1. VAE-encode real image  →  z_0
    2. Add noise at strength σ:  z_s = (1 - σ)·z_0 + σ·ε   (img2img start)
    3. Both branches start from the same z_s  (batch = 2)
         Branch 0: source_prompt  →  provides K at TIER_A layers
         Branch 1: edit_prompt    →  receives K from branch 0 at TIER_A layers
    4. VAE-decode branch 1  →  edited image

Why this works (training-free):
    Branch 0 reconstructs the original image from z_s.
    At TIER_A layers (low-RoPE content-similarity layers), K from branch 0
    is copied into branch 1 — forcing branch 1 to attend to the same spatial
    positions as the reconstruction branch.
    Branch 1's own text conditioning (edit_prompt) drives the semantic change
    through V and Q.  Only K is overridden — structure is preserved,
    content changes.

Usage:
    python NewWork/RealImageStable/run_real_image_stable.py \\
        --hf_token "$HF_TOKEN" \\
        --input          inputs/cat.png \\
        --source_prompt  "a cat" \\
        --prompt         "a cat with blue eyes" \\
        --strength 0.7   \\
        --save_images
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_TIER_A   = frozenset([0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56])
_N_LAYERS = 57


# ──────────────────────── attention processor ─────────────────────────────────

class DualBranchKInjector:
    """
    Drop-in replacement for FluxAttnProcessor2_0.

    At TIER_A layers: copies image-token K from branch 0 (source) into
    branch 1 (edit), locking spatial attention routing so the edit branch
    preserves the original image's structure while edit_prompt drives content.
    """

    def __init__(self):
        self._layer = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):  # noqa: ARG002 (attention_mask unused — FLUX doesn't use it)
        layer       = self._layer
        self._layer = (self._layer + 1) % _N_LAYERS

        B        = hidden_states.shape[0]
        head_dim = attn.inner_dim // attn.heads

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        q = q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # ── K injection at TIER_A layers ──────────────────────────────────────
        if layer in _TIER_A and B >= 2:
            k = k.clone()
            k[1:2, :, txt_len:, :] = k[0:1, :, txt_len:, :].detach()

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if encoder_hidden_states is not None:
            txt_out = attn.to_add_out(out[:, :txt_len])
            img_out = attn.to_out[0](out[:, txt_len:])
            img_out = attn.to_out[1](img_out)
            return img_out, txt_out
        else:
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            return out


# ──────────────────────── helpers ─────────────────────────────────────────────

def _vae_encode(pipe, image: Image.Image, height: int, width: int, device) -> torch.Tensor:
    image  = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr    = np.array(image).astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        z = pipe.vae.encode(tensor).latent_dist.sample()
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return z  # (1, 16, H//8, W//8)


def _pack(z: torch.Tensor) -> torch.Tensor:
    B, C, H, W = z.shape
    return (z.view(B, C, H // 2, 2, W // 2, 2)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, (H // 2) * (W // 2), C * 4))


def _unpack(z: torch.Tensor, height: int, width: int) -> torch.Tensor:
    H, W  = height // 8, width // 8
    B, _, C4 = z.shape
    C = C4 // 4
    return (z.reshape(B, H // 2, W // 2, C, 2, 2)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(B, C, H, W))


def _make_image_ids(h_tokens: int, w_tokens: int, device) -> torch.Tensor:
    ids = torch.zeros(h_tokens, w_tokens, 3)
    ids[..., 1] = torch.arange(h_tokens)[:, None]
    ids[..., 2] = torch.arange(w_tokens)[None, :]
    return ids.reshape(-1, 3).to(device)


def _vae_decode(pipe, z: torch.Tensor) -> Image.Image:
    z_dec = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        img = pipe.vae.decode(z_dec).sample
    img = ((img.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


# ──────────────────────── pipeline ────────────────────────────────────────────

def load_pipeline(model_path, device, hf_token=None, cache_dir=None):
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    ).to(device)
    pipe.set_progress_bar_config(desc="  denoising")
    return pipe


@torch.no_grad()
def run(pipe: FluxPipeline, args):
    input_image = Image.open(args.input).convert("RGB")
    device      = pipe.device
    H, W        = args.height, args.width

    print(f"  input         : {args.input}")
    print(f"  source_prompt : {args.source_prompt}")
    print(f"  edit_prompt   : {args.prompt}")
    print(f"  strength={args.strength}  steps={args.num_steps}  seed={args.seed}")

    # ── 1. Encode text for both branches ─────────────────────────────────────
    src_embeds,  src_pooled,  text_ids = pipe.encode_prompt(
        prompt=args.source_prompt, prompt_2=None,
        device=device, max_sequence_length=512,
    )
    edit_embeds, edit_pooled, _        = pipe.encode_prompt(
        prompt=args.prompt, prompt_2=None,
        device=device, max_sequence_length=512,
    )
    # Batch the two prompt embeddings together
    prompt_embeds = torch.cat([src_embeds,  edit_embeds],  dim=0)  # (2, seq, dim)
    pooled_embeds = torch.cat([src_pooled,  edit_pooled],  dim=0)  # (2, dim)

    # ── 2. VAE-encode real image ──────────────────────────────────────────────
    z_0 = _vae_encode(pipe, input_image, H, W, device)              # (1, 16, H//8, W//8)

    # ── 3. Build mu-shifted sigma schedule ───────────────────────────────────
    out_seq_len   = (H // 16) * (W // 16)
    mu = calculate_shift(
        out_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len",  4096),
        pipe.scheduler.config.get("base_shift",         0.5),
        pipe.scheduler.config.get("max_shift",          1.16),
    )
    linear_sigmas = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps)
    pipe.scheduler.set_timesteps(sigmas=linear_sigmas, mu=mu, device=device)

    all_sigmas    = pipe.scheduler.sigmas.cpu().float().numpy()  # len = num_steps + 1
    all_timesteps = pipe.scheduler.timesteps                     # len = num_steps

    # ── 4. img2img: skip high-noise steps, add noise at start sigma ──────────
    n_skip      = int((1.0 - args.strength) * args.num_steps)
    sigma_start = float(all_sigmas[n_skip])
    timesteps   = all_timesteps[n_skip:]

    g     = torch.Generator(device=device).manual_seed(args.seed)
    noise = randn_tensor(z_0.shape, generator=g, device=device, dtype=z_0.dtype)
    z_s   = (1.0 - sigma_start) * z_0 + sigma_start * noise      # noisy real-image latent

    # ── 5. Shared starting latent for both branches ───────────────────────────
    z_s_packed = _pack(z_s)                                       # (1, seq, 64)
    latents    = z_s_packed.expand(2, -1, -1).clone()             # (2, seq, 64)
    img_ids    = _make_image_ids(H // 16, W // 16, device)

    # ── 6. Install K-injection processor ─────────────────────────────────────
    pipe.transformer.set_attn_processor(DualBranchKInjector())

    # ── 7. Denoising loop ─────────────────────────────────────────────────────
    for t in timesteps:
        timestep = t.expand(2).to(latents.dtype) / 1000.0
        guidance = (torch.full((2,), args.guidance_scale, device=device, dtype=latents.dtype)
                    if args.guidance_scale > 1.0 else None)

        noise_pred = pipe.transformer(
            hidden_states         = latents,
            timestep              = timestep,
            encoder_hidden_states = prompt_embeds,
            pooled_projections    = pooled_embeds,
            txt_ids               = text_ids,
            img_ids               = img_ids,
            guidance              = guidance,
            return_dict           = False,
        )[0]

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # ── 8. Decode edit branch (index 1) ───────────────────────────────────────
    z_edit       = _unpack(latents[1:2], H, W)
    edited_image = _vae_decode(pipe, z_edit)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
        in_path  = os.path.join(args.out_dir, "input.png")
        out_path = os.path.join(args.out_dir, "edited.png")
        input_image.resize((W, H), Image.LANCZOS).save(in_path)
        edited_image.save(out_path)
        print(f"  saved  → {in_path}")
        print(f"  saved  → {out_path}")

    return input_image, edited_image


# ──────────────────────── CLI ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",           required=True,  help="Path to input image")
    p.add_argument("--source_prompt",   required=True,  help="Description of the input image")
    p.add_argument("--prompt",          required=True,  help="Edit instruction / target description")
    p.add_argument("--strength",        type=float, default=0.7,
                   help="Noise level added to real image (0=no change, 1=ignore image). Default 0.7")
    p.add_argument("--num_steps",       type=int,   default=28)
    p.add_argument("--guidance_scale",  type=float, default=3.5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--height",          type=int,   default=1024)
    p.add_argument("--width",           type=int,   default=1024)
    p.add_argument("--model_path",      default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--hf_token",        required=True)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--cache_dir",       default="./models")
    p.add_argument("--out_dir",         default="results/realimageStable")
    p.add_argument("--save_images",     action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n[RealImageStable] Loading {args.model_path} …")
    pipe = load_pipeline(args.model_path, args.device, args.hf_token, args.cache_dir)
    print(f"[RealImageStable] Model loaded.\n")
    run(pipe, args)
    print("[RealImageStable] Done.")
