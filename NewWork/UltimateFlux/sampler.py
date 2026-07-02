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


# ──────────────────── Prompt-to-Prompt colour editing ─────────────────────────

class _KVRecordPolicy(BasePolicy):
    """Pass-1: records K,V at target layers without modifying them."""

    def __init__(self, record_layers: List[int], max_seq_len: int = 512):
        self._layers = set(record_layers)
        self._max_seq_len = max_seq_len
        self.stored: Dict[tuple, tuple] = {}   # (step, layer) → (k, v)

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        if layer in self._layers:
            offset = txt_len if txt_len > 0 else self._max_seq_len
            self.stored[(step, layer)] = (
                k[:, :, offset:, :].detach().clone(),
                v[:, :, offset:, :].detach().clone(),
            )
        return q, k, v


class _P2PColorPolicy(BasePolicy):
    """
    Two-phase P2P injection for colour editing with single-stream structure lock.

    Double-stream layers [0,1,2,4,7,8,9,10,18]:
        Phase 1 (step < anchor_end_frac × n_steps): K+V → aligns Q_edit ≈ Q_src
        Phase 2 (remaining steps):                  K-only → coarse identity held

    Single-stream TIER_A layers [25,28,37,42,45,50,56]:
        All steps: K-only → fine-detail rendering locked (plates, facial edges)
        V is never injected here → V from the edit branch carries the new colour

    Why single-stream K injection is needed:
        Single-stream blocks do the final fine-detail rendering pass.  Without K
        injection there, spatial structure at that level (license-plate text, fine
        face contours) drifts freely even when coarse composition is held by the
        double-stream blocks.  K-only keeps WHERE those blocks attend without
        freezing WHAT colour they render.
    """
    # 9 double-stream layers: TIER_A∩DS ∪ {1,2,4}
    _DOUBLE_INJECT: List[int] = sorted(
        {l for l in TIER_A if l < N_DOUBLE} | {1, 2, 4}
    )
    # 7 single-stream layers: TIER_A∩SS
    _SINGLE_INJECT: frozenset = frozenset(l for l in TIER_A if l >= N_DOUBLE)
    # 16 layers to record in Pass 1
    _RECORD_LAYERS: List[int] = sorted(set(TIER_A) | {1, 2, 4})

    def __init__(
        self,
        stored: Dict[tuple, tuple],
        max_seq_len: int = 512,
        anchor_end_frac: float = 0.25,
    ):
        self._stored = stored
        self._max_seq_len = max_seq_len
        self._anchor = anchor_end_frac
        self._double_set = frozenset(self._DOUBLE_INJECT)

    def inject_qkv(self, q, k, v, layer, step, n_steps, txt_len=0):
        is_double = layer in self._double_set
        is_single = layer in self._SINGLE_INJECT
        if not is_double and not is_single:
            return q, k, v

        entry = self._stored.get((step, layer))
        if entry is None:
            return q, k, v
        stored_k, stored_v = entry

        offset = txt_len if txt_len > 0 else self._max_seq_len
        k = k.clone()
        k[:, :, offset:, :] = stored_k          # always: K locks WHERE to attend

        if is_double and step < int(self._anchor * n_steps):
            # Phase 1 (double-stream only): K+V aligns edit branch to source
            v = v.clone()
            v[:, :, offset:, :] = stored_v

        # Phase 2 / single-stream: V from edit branch → carries new colour
        return q, k, v


@torch.no_grad()
def generate_p2p(
    pipe: FluxPipeline,
    source_prompt: str,
    edit_prompt: str,
    inject_layers: List[int],          # ignored; kept for API compat
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    anchor_end_frac: float = 0.25,
    freq_sigma: float = 0.0,           # ignored; kept for API compat
    img2img_strength: float = 0.6,     # ignored; kept for API compat
) -> Tuple[Image.Image, Image.Image]:
    """
    Prompt-to-Prompt colour editing: same z_T, two-phase DS injection + SS K-lock.

    Pass 1: generate source from source_prompt + seed, record K,V at 16 layers
            (9 double-stream TIER_A+HOTSPOT + 7 single-stream TIER_A).
    Pass 2: denoise from the same z_T with edit_prompt:
        Double-stream (9 layers):
            Phase 1 (steps 0 .. anchor_end_frac×n): K+V → Q_edit → Q_src
            Phase 2 (remaining steps):               K-only → coarse identity
        Single-stream (7 layers):
            All steps: K-only → fine-detail structure locked (plates, faces)
            V always from edit → colour flows through unimpeded

    anchor_end_frac
        0.15 → more colour change | 0.25 → balanced (default) | 0.40 → more identity

    Returns (src_img, edit_img).
    """
    exec_device = getattr(pipe, "_execution_device", device)
    num_ch = pipe.transformer.config.in_channels // 4
    lat_h  = height // 8
    lat_w  = width  // 8

    _g   = torch.Generator(device=exec_device).manual_seed(seed)
    _one = randn_tensor(
        (1, num_ch, lat_h, lat_w),
        generator=_g, device=exec_device, dtype=torch.bfloat16,
    )
    latent = (
        _one.view(1, num_ch, lat_h // 2, 2, lat_w // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(1, (lat_h // 2) * (lat_w // 2), num_ch * 4)
    )

    def _run(prompt: str, policy: BasePolicy) -> Image.Image:
        proc = UltimateFluxAttnProcessor(
            policy=policy, total_layers=N_LAYERS, total_steps=num_steps
        )
        proc.reset()
        pipe.transformer.set_attn_processor(proc)
        return pipe(
            prompt=prompt,
            latents=latent.clone(),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            output_type="pil",
        ).images[0]

    # Pass 1 — source image + record K,V at 16 layers (9 DS + 7 SS)
    print("[UltimateFlux P2P] Pass 1: generating source, recording K,V at "
          f"{len(_P2PColorPolicy._RECORD_LAYERS)} layers…")
    rec = _KVRecordPolicy(_P2PColorPolicy._RECORD_LAYERS, max_sequence_length)
    src_img = _run(source_prompt, rec)

    # Pass 2 — colour edit: DS two-phase + SS K-only
    anchor_steps = int(anchor_end_frac * num_steps)
    n_ds = len(_P2PColorPolicy._DOUBLE_INJECT)
    n_ss = len(_P2PColorPolicy._SINGLE_INJECT)
    print(
        f"[UltimateFlux P2P] Pass 2: "
        f"DS 0-{anchor_steps-1} K+V | DS {anchor_steps}-{num_steps-1} K-only | "
        f"SS all-steps K-only | ({n_ds} DS + {n_ss} SS layers)…"
    )
    inj = _P2PColorPolicy(
        rec.stored,
        max_seq_len=max_sequence_length,
        anchor_end_frac=anchor_end_frac,
    )
    edit_img = _run(edit_prompt, inj)

    return src_img, edit_img
