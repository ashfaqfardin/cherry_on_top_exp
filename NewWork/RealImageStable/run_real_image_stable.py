"""
RealImageStable — Training-Free Real Image Editing on FLUX.1-dev
             via RF-Edit (Rectified Flow Inversion + V-Feature Injection)

Based on: "Taming Rectified Flow for Inversion and Editing" (RF-Solver / RF-Edit, ICML 2025)
          arXiv 2411.04746  |  github.com/wangjiangshan0725/RF-Solver-Edit

Pipeline:
    1. VAE-encode real image  →  z_0  (clean latent, σ = 0)
    2. Inversion:  run FLUX forward with source_prompt
                   Euler integration: z_0  →  z_N  (noisy latent, σ = σ_max)
                   At the LAST inversion step: capture V (value) features
                   from every self-attention layer.
    3. Editing:    run FLUX backward with edit_prompt  (standard generation)
                   At the FIRST denoising step: inject saved V features
                   into every self-attention layer.
                   Remaining steps run freely under edit_prompt.
    4. VAE-decode  →  edited image.

Why V injection works:
    V features encode WHAT the model attends to (content/texture at each position).
    Injecting V from the source pass at the first denoising step (highest noise)
    stamps a coarse structural scaffold onto the edit trajectory — background,
    composition, lighting — without locking fine detail.  The edit_prompt then
    drives semantic changes through Q and the model's free later steps.

Usage:
    python NewWork/RealImageStable/run_real_image_stable.py \\
        --hf_token "$HF_TOKEN" \\
        --input          inputs/light.png \\
        --source_prompt  "Glowing marquee GLOW sign on a brick wall." \\
        --prompt         "Glowing marquee FLUX sign on a brick wall." \\
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
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_N_LAYERS = 57


# ──────────────────────── attention processor ─────────────────────────────────

class RFEditProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 / FluxSingleAttnProcessor2_0.

    mode='pass'    — normal attention, no capture or injection.
    mode='capture' — additionally stores V (image-token portion) for every layer.
    mode='inject'  — replaces the image-token portion of V with stored values.

    Works for both double-stream blocks (encoder_hidden_states is not None)
    and single-stream blocks (encoder_hidden_states is None, text+image combined).
    """

    def __init__(self):
        self._layer   = 0
        self._txt_seq = None   # text sequence length, cached from double-stream call
        self.mode     = 'pass'
        self._stored_v: dict = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
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
            # ── double-stream block ───────────────────────────────────────────
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            self._txt_seq = txt_len
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)
        else:
            # ── single-stream block: hidden_states = [txt | img] ─────────────
            txt_len = self._txt_seq or 0

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # ── capture / inject V (image-token portion only) ─────────────────────
        if self.mode == 'capture':
            self._stored_v[layer] = v[:, :, txt_len:, :].detach().clone()

        elif self.mode == 'inject' and layer in self._stored_v:
            stored = self._stored_v[layer].to(v.device, dtype=v.dtype)
            v = v.clone()
            v[:, :, txt_len:, :] = stored

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if encoder_hidden_states is not None:
            txt_out = attn.to_add_out(out[:, :txt_len])
            img_out = attn.to_out[0](out[:, txt_len:])
            img_out = attn.to_out[1](img_out)
            return img_out, txt_out
        else:
            if hasattr(attn, "to_out") and attn.to_out is not None:
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
    H, W     = height // 8, width // 8
    B, _, C4 = z.shape
    C        = C4 // 4
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


def _transformer_step(pipe, z, sigma, prompt_embeds, pooled_embeds,
                      text_ids, img_ids, guidance_scale, device):
    """Single FLUX transformer forward pass. sigma in [0, 1]."""
    timestep = torch.tensor([sigma], device=device, dtype=z.dtype)
    guidance  = torch.full((1,), guidance_scale, device=device, dtype=z.dtype)
    return pipe.transformer(
        hidden_states         = z,
        timestep              = timestep,
        encoder_hidden_states = prompt_embeds,
        pooled_projections    = pooled_embeds,
        txt_ids               = text_ids,
        img_ids               = img_ids,
        guidance              = guidance,
        return_dict           = False,
    )[0]


# ──────────────────────── pipeline ────────────────────────────────────────────

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
    device      = pipe.device
    H, W        = args.height, args.width

    print(f"  input         : {args.input}")
    print(f"  source_prompt : {args.source_prompt}")
    print(f"  edit_prompt   : {args.prompt}")
    print(f"  steps={args.num_steps}  inject_steps={args.inject_steps}  "
          f"guidance={args.guidance_scale}  inv_guidance={args.inv_guidance}")

    # ── 1. Encode prompts ─────────────────────────────────────────────────────
    src_embeds,  src_pooled,  text_ids = pipe.encode_prompt(
        prompt=args.source_prompt, prompt_2=None,
        device=device, max_sequence_length=512,
    )
    edit_embeds, edit_pooled, _        = pipe.encode_prompt(
        prompt=args.prompt, prompt_2=None,
        device=device, max_sequence_length=512,
    )

    # ── 2. VAE-encode real image ──────────────────────────────────────────────
    z_0     = _vae_encode(pipe, input_image, H, W, device)   # (1, 16, H//8, W//8)
    img_ids = _make_image_ids(H // 16, W // 16, device)

    # ── 3. Build mu-shifted sigma schedule ───────────────────────────────────
    out_seq_len = (H // 16) * (W // 16)
    mu = calculate_shift(
        out_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len",  4096),
        pipe.scheduler.config.get("base_shift",         0.5),
        pipe.scheduler.config.get("max_shift",          1.16),
    )
    linear_sigmas = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps)
    pipe.scheduler.set_timesteps(sigmas=linear_sigmas, mu=mu, device=device)

    # gen_sigmas: [σ_max, σ_{N-1}, …, σ_1, 0]  (N+1 values, decreasing)
    gen_sigmas    = pipe.scheduler.sigmas.cpu().float().numpy()
    edit_timesteps = pipe.scheduler.timesteps   # [t_max, …, t_1] in [0, 1000]

    # inv_sigmas: [0, σ_1, …, σ_{N-1}, σ_max]  (reversed, increasing)
    inv_sigmas = gen_sigmas[::-1]               # N+1 values

    # ── 4. Install processor ──────────────────────────────────────────────────
    processor = RFEditProcessor()
    pipe.transformer.set_attn_processor(processor)

    # ── 5. Inversion  (z_0 → z_noisy)  Euler, source_prompt ─────────────────
    print("  inverting …")
    z = _pack(z_0)
    for i in range(args.num_steps):
        sigma_c = float(inv_sigmas[i])
        sigma_n = float(inv_sigmas[i + 1])
        dt      = sigma_n - sigma_c                      # > 0 (moving toward noise)

        processor.mode  = 'capture' if i == args.num_steps - 1 else 'pass'
        processor._layer = 0

        v = _transformer_step(pipe, z, sigma_c,
                              src_embeds, src_pooled, text_ids, img_ids,
                              args.inv_guidance, device)
        z = z + dt * v                                   # Euler inversion step

    z_noisy = z.clone()
    print(f"  inversion done — {len(processor._stored_v)} layer V-features saved.")

    # ── 6. Editing  (z_noisy → z_edit)  edit_prompt, V injection at step 0 ──
    print("  editing …")
    z = z_noisy
    for i, t in enumerate(edit_timesteps):
        processor.mode  = 'inject' if i < args.inject_steps else 'pass'
        processor._layer = 0

        sigma = float(t.item()) / 1000.0                # scheduler t → [0,1]
        v = _transformer_step(pipe, z, sigma,
                              edit_embeds, edit_pooled, text_ids, img_ids,
                              args.guidance_scale, device)

        z = pipe.scheduler.step(v, t, z, return_dict=False)[0]

    # ── 7. Decode ─────────────────────────────────────────────────────────────
    z_edit       = _unpack(z, H, W)
    edited_image = _vae_decode(pipe, z_edit)

    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
        input_image.resize((W, H), Image.LANCZOS).save(
            p := os.path.join(args.out_dir, "input.png"))
        edited_image.save(
            q := os.path.join(args.out_dir, "edited.png"))
        print(f"  saved → {p}")
        print(f"  saved → {q}")

    return input_image, edited_image


# ──────────────────────── CLI ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Training-free real image editing on FLUX.1-dev via RF-Edit."
    )
    p.add_argument("--input",          required=True,
                   help="Path to the input image.")
    p.add_argument("--source_prompt",  required=True,
                   help="Text description of the INPUT image (used for inversion).")
    p.add_argument("--prompt",         required=True,
                   help="Target description for the EDIT (used for generation).")
    p.add_argument("--num_steps",      type=int,   default=28,
                   help="Denoising / inversion steps. Default 28.")
    p.add_argument("--inject_steps",   type=int,   default=1,
                   help="Number of denoising steps that receive V injection. "
                        "1 = first step only (recommended, per RF-Edit paper). "
                        "Higher values increase structural preservation but "
                        "reduce edit freedom. Default 1.")
    p.add_argument("--guidance_scale", type=float, default=3.5,
                   help="CFG strength for the editing pass. Default 3.5.")
    p.add_argument("--inv_guidance",   type=float, default=1.0,
                   help="CFG strength for the inversion pass. Default 1.0 (no boost).")
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
    print("[RealImageStable] Model loaded.\n")
    run(pipe, args)
    print("[RealImageStable] Done.")
