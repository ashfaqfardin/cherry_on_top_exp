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


# ──────────────────── Latent-space colour editing ─────────────────────────────

def _latent_blend(
    z_src: torch.Tensor,
    z_edit: torch.Tensor,
    top_k: int = 0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Colour replacement in FLUX packed latent space — no pixel-space ops.

    Both z_src and z_edit are generated from the same seed (same z_T), so
    composition is nearly identical.  The colour prompt drives most of the
    latent delta.

    Soft per-token mask (the key insight):
        w[i] = ‖delta[i]‖₂ / p95(‖delta‖₂)   clipped to [0, 1]
        Tokens in the coloured region (car body, hair) changed the most
        between source and edit → w ≈ 1 → new colour applied fully.
        Tokens that barely changed (plates, background, face) → w ≈ 0
        → z_src preserved → structure intact.

    Optional SVD low-rank filter (top_k > 0):
        Replace delta with its rank-k approximation before masking.
        Top-k singular vectors capture the dominant GLOBAL colour shift
        in channel space; local structural drift (which is lower-rank)
        is discarded.  Start with top_k=0; increase if colour is weak.

    z_src, z_edit : (1, N, C) FLUX packed latents  (bfloat16 or float16)
    top_k         : 0 = no SVD  |  1–8 = keep top-k singular vectors
    alpha         : delta scale (1.0 = full swap | >1 = amplify colour)
    """
    delta = z_edit.float() - z_src.float()    # (1, N, C)

    if top_k > 0:
        d        = delta[0]                                       # (N, C)
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        k        = min(top_k, S.shape[0])
        delta    = ((U[:, :k] * S[:k]) @ Vh[:k, :]).unsqueeze(0) # (1, N, C)

    # Robust per-token mask: use 95th-percentile as scale so one outlier
    # token doesn't compress all others to near-zero weight.
    norms = delta[0].norm(dim=-1)                                 # (N,)
    scale = torch.quantile(norms, 0.95).clamp(min=1e-6)
    w     = (norms / scale).clamp(max=1.0)                       # (N,) ∈ [0, 1]

    z_out = z_src.float() + alpha * w.unsqueeze(-1).unsqueeze(0) * delta
    return z_out.to(z_src.dtype)


def _decode_latents(
    pipe: FluxPipeline,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    """Unpack FLUX packed latents and decode via the VAE → PIL image."""
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
    inject_layers: List[int],          # unused; kept for API compat
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    anchor_end_frac: float = 0.0,      # unused; kept for API compat
    freq_sigma: float = 0.0,           # unused; kept for API compat
    img2img_strength: float = 0.6,     # unused; kept for API compat
    source_color: str = "",            # unused; kept for API compat
    target_color: str = "",            # unused; kept for API compat
    edge_strength: float = 0.7,        # unused; kept for API compat
    latent_top_k: int = 0,
    latent_alpha: float = 1.0,
) -> Tuple[Image.Image, Image.Image]:
    """
    Colour editing via latent-space delta blending — two FLUX passes, no CV.

    Pass 1 — pipe(source_prompt, seed) → z_src  (packed latent, skip VAE decode)
    Pass 2 — pipe(edit_prompt,   seed) → z_edit (packed latent, same z_T)

    Both passes start from the same noise z_T (same generator seed), so the
    scene composition is nearly identical.  The colour change in the edit
    prompt drives a large, spatially concentrated delta in the latent.

    Blend (pure latent arithmetic):
        delta    = z_edit − z_src
        [SVD]    delta ← rank-k(delta)       if latent_top_k > 0
        w[i]     = ‖delta[i]‖ / p95‖delta‖  per-token soft mask
        z_result = z_src + alpha × w × delta

    Why license plates / face are preserved:
        Those regions are the same colour in both passes (they're not the
        car body / hair) → delta ≈ 0 there → w ≈ 0 → z_src preserved.

    latent_top_k (default 0 — try 4 if colour is weak):
        SVD extracts the dominant global colour direction in channel space.
        Discards local structural drift that crept in from prompt wording.

    latent_alpha (default 1.0 — try 1.2 to amplify):
        Scales the colour delta.  >1 overshoots slightly, giving a stronger
        colour impression (useful when the delta is subtle).

    Returns (src_img, edit_img).
    """
    exec_device = getattr(pipe, "_execution_device", device)

    def _run_latent(prompt: str) -> torch.Tensor:
        return pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator(device=exec_device).manual_seed(seed),
            output_type="latent",
        ).images                              # (1, N, C) packed latent tensor

    print("[UltimateFlux P2P] Pass 1: generating source latent…")
    z_src = _run_latent(source_prompt)

    print("[UltimateFlux P2P] Pass 2: generating edit latent (same seed)…")
    z_edit = _run_latent(edit_prompt)

    print(
        f"[UltimateFlux P2P] Blending in latent space "
        f"(top_k={latent_top_k}, alpha={latent_alpha})…"
    )
    z_blend = _latent_blend(z_src, z_edit, top_k=latent_top_k, alpha=latent_alpha)

    src_img  = _decode_latents(pipe, z_src,   height, width)
    edit_img = _decode_latents(pipe, z_blend, height, width)
    return src_img, edit_img
