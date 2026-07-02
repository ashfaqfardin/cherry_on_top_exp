"""
Shared dual-branch FLUX.1-dev sampler for UltimateFlux.

All tasks share a B=2 denoising loop:
  batch index 0 = source/content branch  (unmodified)
  batch index 1 = edit/styled branch     (receives injections from policy)

InjectionPolicy controls Q/K/V manipulation and block-level hooks at each (layer, step).

Layer index convention (combined 0-based across all 57 blocks):
  double-stream:  0–18  → pipe.transformer.transformer_blocks[i]
  single-stream: 19–56  → pipe.transformer.single_transformer_blocks[i-19]
"""

import os
import sys
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from typing import Optional, List, Tuple, Dict, Callable

# ── Layer tier constants (from FreeFlux ICCV 2025, validated on FLUX.1-dev) ──
# Tier A — Content-similarity-dependent layers: appearance, style, texture, object
# Source: FreeFlux non_rigid_attn_utils.py layer_idx (RoPE frequency analysis)
TIER_A = [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]

# Tier B — All layers: used when shape/pose deformation is needed
TIER_B = list(range(57))

N_DOUBLE = 19
N_SINGLE = 38
N_LAYERS = N_DOUBLE + N_SINGLE  # 57


# ─────────────────────────────── Base Policy ─────────────────────────────────

class BasePolicy:
    """
    Abstract injection policy.  Subclasses override the three entry points.
    """

    def inject_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        step: int,
        n_steps: int,
        txt_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Called inside every attention layer.  Modify q/k/v for edit branch."""
        return q, k, v

    def get_block_hooks(self) -> Dict[int, Callable]:
        """
        Return {block_idx: hook_fn} for transformer block forward hooks.
        Hooks are registered before generation and removed after.
        block_idx uses the same combined 0-56 convention as layer tiers.
        """
        return {}

    def pre_generate(self, pipe: FluxPipeline, **kwargs) -> None:
        """Called once before the denoising loop (style extraction, inversion, …)."""
        pass

    def post_process(self, src_img: Image.Image, edit_img: Image.Image) -> Image.Image:
        """Called after generation with the decoded PIL images.  Default: no-op."""
        return edit_img


# ─────────────────────────── Attention Processor ─────────────────────────────

class UltimateFluxAttnProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 that delegates Q/K/V
    manipulation to a pluggable BasePolicy.

    Maintains a (cur_step, cur_att_layer) counter identical to FreeFlux's
    processors so policies can gate on specific (layer, step) pairs.
    """

    def __init__(self, policy: BasePolicy, total_layers: int = N_LAYERS, total_steps: int = 4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("UltimateFluxAttnProcessor requires PyTorch 2.0+")
        self.policy = policy
        self.total_layers = total_layers
        self.total_steps = total_steps
        self.cur_step = 0
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def _tick(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.total_layers:
            self.cur_att_layer = 0
            self.cur_step = (self.cur_step + 1) % self.total_steps

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        out = self._forward(attn, hidden_states, encoder_hidden_states,
                            attention_mask, image_rotary_emb)
        self._tick()
        return out

    def _forward(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask, image_rotary_emb):
        layer = self.cur_att_layer
        step  = self.cur_step
        is_double = encoder_hidden_states is not None
        batch_size = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        inner_dim = k.shape[-1]
        head_dim  = inner_dim // attn.heads

        q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]          # number of text tokens in the prefix
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # ── Policy injection ──────────────────────────────────────────────
        q, k, v = self.policy.inject_qkv(q, k, v, layer, step, self.total_steps, txt_len)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out

        return out


# ────────────────────────── Pipeline helpers ──────────────────────────────────

def load_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-dev",
    hf_token: Optional[str] = None,
    device: str = "cuda",
    cpu_offload: bool = False,
    cache_dir: str = "./models",
) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


@torch.no_grad()
def generate_dual_branch(
    pipe: FluxPipeline,
    policy: BasePolicy,
    source_prompt: str,
    edit_prompt: str,
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
) -> Tuple[Image.Image, Image.Image]:
    """
    Dual-branch denoising with injections from policy.

    Returns (source_image, edit_image).  source_image is the unmodified
    reconstruction; edit_image is the policy-steered output.
    """
    # 1. Policy pre-hook (style extraction, reasoning pass, inversion, etc.)
    policy.pre_generate(
        pipe,
        device=device,
        height=height,
        width=width,
        num_steps=num_steps,
        seed=seed,
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
    )

    # 2. Shared initial latent — both branches MUST start from identical noise.
    #    Without this, pipe(prompt=[A, B]) generates different noise for A and B
    #    (sequential samples from the same generator), so K,V injection between
    #    branches fights completely different initial conditions and produces garbage.
    #    FreeFlux's run_non_rigid.py does exactly this: generate one latent, expand.
    exec_device = getattr(pipe, '_execution_device', device)
    num_ch  = pipe.transformer.config.in_channels // 4   # 16 for FLUX
    lat_h   = height // 8
    lat_w   = width  // 8
    _g      = torch.Generator(device=exec_device).manual_seed(seed)
    _one    = randn_tensor(
        (1, num_ch, lat_h, lat_w),
        generator=_g,
        device=exec_device,
        dtype=torch.bfloat16,
    )
    # Pack: (B, C, lat_h, lat_w) → (B, lat_h//2 * lat_w//2, C*4)
    # This is the standard FLUX packing used inside prepare_latents.
    shared_latents = (
        _one.expand(2, -1, -1, -1).clone()
        .view(2, num_ch, lat_h // 2, 2, lat_w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(2, (lat_h // 2) * (lat_w // 2), num_ch * 4)
    )

    # 3. Install custom attention processor
    proc = UltimateFluxAttnProcessor(
        policy=policy, total_layers=N_LAYERS, total_steps=num_steps
    )
    proc.reset()
    pipe.transformer.set_attn_processor(proc)

    # 4. Register block-level forward hooks
    handles: List = []
    for block_idx, hook_fn in policy.get_block_hooks().items():
        if block_idx < N_DOUBLE:
            block = pipe.transformer.transformer_blocks[block_idx]
        else:
            block = pipe.transformer.single_transformer_blocks[block_idx - N_DOUBLE]
        handles.append(block.register_forward_hook(hook_fn))

    try:
        result = pipe(
            prompt=[source_prompt, edit_prompt],
            latents=shared_latents,        # pre-packed, shared across both branches
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            output_type="pil",
        )
    finally:
        for h in handles:
            h.remove()

    src_img  = result.images[0]
    edit_img = policy.post_process(src_img, result.images[1])
    return src_img, edit_img




# ──────────────────── Split-denoising colour editing ──────────────────────────

def _decode_latents(
    pipe: FluxPipeline,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    """Unpack FLUX packed latents and decode via the VAE -> PIL image."""
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    latents   = pipe._unpack_latents(packed_latents, height, width, vae_scale)
    latents   = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    raw       = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(raw, output_type="pil")[0]


@torch.no_grad()
def generate_p2p(
    pipe: FluxPipeline,
    source_prompt: str,
    edit_prompt: str,
    inject_layers: List[int],           # unused; kept for API compat
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    anchor_end_frac: float = 0.0,       # unused; kept for API compat
    freq_sigma: float = 0.0,            # unused; kept for API compat
    img2img_strength: float = 0.6,      # unused; kept for API compat
    source_color: str = "",             # unused; kept for API compat
    target_color: str = "",             # unused; kept for API compat
    edge_strength: float = 0.7,         # unused; kept for API compat
    latent_top_k: int = 0,              # unused; kept for API compat
    latent_alpha: float = 1.0,          # unused; kept for API compat
    color_structure_frac: float = 0.4,
) -> Tuple[Image.Image, Image.Image]:
    """
    Colour editing via split-denoising: shared structure phase -> diverged colour phase.

    The num_steps denoising steps are split at N = round(color_structure_frac * num_steps):

    Phase 1  steps [0, N)   -- source prompt, single branch:
        Both source and edit will share this denoising prefix.
        Early steps commit the global composition -- car position, plate
        location, face layout -- as a joint latent z_N.

    Phase 2  steps [N, end) -- two branches, batched in one transformer call:
        Source branch: continues with source_prompt -> renders red car.
        Edit   branch: starts from z_N, uses edit_prompt -> renders blue car.
        Because both branches are initialised from the IDENTICAL z_N, the
        structure is locked by construction. No blending, no CV, no injection.

    This eliminates the "overlap of two images" artifact that comes from
    blending independently-denoised latents: both branches share the same
    trajectory up to step N, guaranteeing structural alignment.

    color_structure_frac (tune in JSON / CLI):
        0.2  ~5 shared steps   more colour freedom, slight drift risk
        0.4  ~11 shared steps  good balance (default)
        0.6  ~17 shared steps  very strong structure lock

    Returns (src_img, edit_img).
    """
    exec_device = getattr(pipe, "_execution_device", device)
    split_step  = max(1, int(color_structure_frac * num_steps))

    # -- Encode both prompts --------------------------------------------------
    def _encode(prompt: str):
        enc = pipe.encode_prompt(
            prompt, prompt_2=None, device=exec_device,
            num_images_per_prompt=1, max_sequence_length=max_sequence_length,
        )
        # Recent diffusers: (pe, pp, txt_ids).  Older: 6 values.
        if len(enc) == 3:
            pe, pp, txt_ids = enc
        else:
            pe, _, pp, _, txt_ids, _ = enc
        return pe, pp, txt_ids

    src_pe,  src_pp,  src_txt  = _encode(source_prompt)
    edit_pe, edit_pp, edit_txt = _encode(edit_prompt)

    # -- Initial packed latent z_T --------------------------------------------
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    num_ch    = pipe.transformer.config.in_channels // 4   # VAE channels = 16
    lat_h     = 2 * (height // (vae_scale * 2))           # unpacked latent H
    lat_w     = 2 * (width  // (vae_scale * 2))           # unpacked latent W

    gen   = torch.Generator(device=exec_device).manual_seed(seed)
    raw_z = randn_tensor(
        (1, num_ch, lat_h, lat_w),
        generator=gen, device=exec_device, dtype=src_pe.dtype,
    )
    z_T     = pipe._pack_latents(raw_z, 1, num_ch, lat_h, lat_w)   # (1, N, 64)
    img_ids = pipe._prepare_latent_image_ids(
        1, lat_h // 2, lat_w // 2, exec_device, src_pe.dtype,
    )                                                                 # (1, N, 3)

    # -- Timesteps: match pipe()'s schedule -----------------------------------
    # FLUX uses dynamic shifting: mu scales the schedule to image resolution.
    image_seq_len = (lat_h // 2) * (lat_w // 2)
    sched_cfg     = pipe.scheduler.config
    if getattr(sched_cfg, "use_dynamic_shifting", False):
        base_seq  = getattr(sched_cfg, "base_image_seq_len", 256)
        max_seq   = getattr(sched_cfg, "max_image_seq_len",  4096)
        base_shft = getattr(sched_cfg, "base_shift",         0.5)
        max_shft  = getattr(sched_cfg, "max_shift",          1.15)
        m   = (max_shft - base_shft) / (max_seq - base_seq)
        mu  = image_seq_len * m + (base_shft - m * base_seq)
        pipe.scheduler.set_timesteps(num_steps, device=exec_device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=exec_device)
    timesteps = pipe.scheduler.timesteps                              # length = num_steps

    has_guidance = getattr(pipe.transformer.config, "guidance_embeds", False)

    def _transformer_step(
        lat: torch.Tensor,
        t: torch.Tensor,
        pe: torch.Tensor,
        pp: torch.Tensor,
        txt: torch.Tensor,
    ) -> torch.Tensor:
        B  = lat.shape[0]
        ts = t.expand(B)
        g  = (torch.full([B], guidance_scale, device=exec_device, dtype=lat.dtype)
              if has_guidance else None)
        noise = pipe.transformer(
            hidden_states=lat,
            timestep=ts / 1000.0,
            guidance=g,
            pooled_projections=pp,
            encoder_hidden_states=pe,
            txt_ids=txt,
            img_ids=img_ids.expand(B, -1, -1),
            return_dict=False,
        )[0]
        return pipe.scheduler.step(noise, t, lat, return_dict=False)[0]

    # -- Phase 1: single branch with source prompt ----------------------------
    print(f"[UltimateFlux P2P] Phase 1: {split_step} shared steps (source prompt)...")
    latents = z_T.clone()
    for t in timesteps[:split_step]:
        latents = _transformer_step(latents, t, src_pe, src_pp, src_txt)

    z_split  = latents.clone()   # structure checkpoint -- both branches fork here
    lat_src  = z_split.clone()
    lat_edit = z_split.clone()

    # -- Phase 2: two branches batched into one transformer call per step -----
    print(
        f"[UltimateFlux P2P] Phase 2: {num_steps - split_step} diverged steps "
        f"(source | edit batched)..."
    )
    for t in timesteps[split_step:]:
        # Concatenate both branches -- one forward pass handles both.
        lat_b  = torch.cat([lat_src, lat_edit])                  # (2, N, 64)
        pe_b   = torch.cat([src_pe,  edit_pe])
        pp_b   = torch.cat([src_pp,  edit_pp])
        txt_b  = torch.cat([src_txt, edit_txt])
        ts     = t.expand(2)
        g      = (torch.full([2], guidance_scale, device=exec_device, dtype=lat_b.dtype)
                  if has_guidance else None)

        noise_b = pipe.transformer(
            hidden_states=lat_b,
            timestep=ts / 1000.0,
            guidance=g,
            pooled_projections=pp_b,
            encoder_hidden_states=pe_b,
            txt_ids=txt_b,
            img_ids=img_ids.expand(2, -1, -1),
            return_dict=False,
        )[0]

        noise_src, noise_edit = noise_b.chunk(2)
        lat_src  = pipe.scheduler.step(noise_src,  t, lat_src,  return_dict=False)[0]
        lat_edit = pipe.scheduler.step(noise_edit, t, lat_edit, return_dict=False)[0]

    src_img  = _decode_latents(pipe, lat_src,  height, width)
    edit_img = _decode_latents(pipe, lat_edit, height, width)
    return src_img, edit_img
