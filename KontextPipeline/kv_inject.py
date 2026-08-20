"""
kv_inject.py — Object appearance preservation via K/V injection with DAAM spatial masking.

Two-phase denoising:

  Phase 1 (steps 0 → warm_steps):
    Global K/V injection at all gen tokens + accumulate DAAM heatmap.
    During these steps, Q and K are captured from one single-stream TIER_A layer.
    The gen→obj-word attention energy gives a spatial activation map.

  Phase 2 (steps warm_steps → end):
    DAAM heatmap thresholded → binary token mask.
    K/V injection only at masked (object) tokens.
    Background tokens receive no injection.

Injection formula (at each masked token position):
    K_gen[tok] ← (1 - α_K) * K_gen[tok] + α_K * K_obj[tok]
    V_gen[tok] ← (1 - α_V) * V_gen[tok] + α_V * V_obj[tok]

    α_K = 0.85 (hard attention routing)
    α_V = 0.50 (soft content contribution)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from .flow_inject import (
    _pack, _unpack, _image_ids, _encode_image, _compute_scheduler_mu,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_TIER_A         = frozenset({0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56})
_N_DOUBLE       = 19
_SINGLE_TXT_LEN = 512

_TIER_A_DOUBLE  = frozenset(i             for i in _TIER_A if i < _N_DOUBLE)
_TIER_A_SINGLE  = frozenset(i - _N_DOUBLE for i in _TIER_A if i >= _N_DOUBLE)

# Single-stream layer used for DAAM (layer 28 global = single_idx 9)
_DAAM_SINGLE_IDX = 28 - _N_DOUBLE   # = 9


# ── K/V capture from obj_img ──────────────────────────────────────────────────

class _KVCapture:
    def __init__(self, n_tok: int):
        self.n_tok  = n_tok
        self.kv: Dict[int, Dict[str, torch.Tensor]] = {}
        self._hooks = []

    def register(self, pipe):
        for i in _TIER_A_DOUBLE:
            b = pipe.transformer.transformer_blocks[i]
            self._add(b.attn.to_k, i, "k")
            self._add(b.attn.to_v, i, "v")
        for j in _TIER_A_SINGLE:
            b = pipe.transformer.single_transformer_blocks[j]
            self._add(b.attn.to_k, _N_DOUBLE + j, "k")
            self._add(b.attn.to_v, _N_DOUBLE + j, "v")

    def _add(self, module, layer_idx: int, kv_type: str):
        n = self.n_tok
        def hook(mod, inp, out):
            if layer_idx not in self.kv:
                self.kv[layer_idx] = {}
            self.kv[layer_idx][kv_type] = out[:, :n, :].detach().cpu()
        self._hooks.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── DAAM heatmap accumulator ──────────────────────────────────────────────────

class _DAAMAccumulator:
    """
    Hooks to_q / to_k at one single-stream TIER_A layer.
    After each forward pass, accumulates gen→obj-word attention energy.
    After warm_steps calls, compute_mask() returns a (n_tok,) bool tensor.
    """

    def __init__(self, n_tok: int, n_heads: int, d_head: int):
        self.n_tok   = n_tok
        self.n_heads = n_heads
        self.d_head  = d_head
        self._q      = None
        self._k      = None
        self._heatmap: Optional[torch.Tensor] = None
        self._count  = 0
        self._hooks  = []

    def register(self, pipe):
        blk = pipe.transformer.single_transformer_blocks[_DAAM_SINGLE_IDX]
        self._hooks.append(blk.attn.to_q.register_forward_hook(self._hook_q))
        self._hooks.append(blk.attn.to_k.register_forward_hook(self._hook_k))

    def _hook_q(self, mod, inp, out):
        self._q = out.detach().cpu()

    def _hook_k(self, mod, inp, out):
        self._k = out.detach().cpu()
        self._step()

    def _step(self):
        if self._q is None or self._k is None:
            return
        n = self.n_tok
        H = self.n_heads
        d = self.d_head
        B, seq, _ = self._q.shape

        q = self._q.view(B, seq, H, d).permute(0, 2, 1, 3).float()  # (B,H,seq,d)
        k = self._k.view(B, seq, H, d).permute(0, 2, 1, 3).float()

        q_gen = q[:, :, :n,    :]   # (B, H, n_tok, d)   — gen image tokens
        k_txt = k[:, :, 2*n:,  :]   # (B, H, txt_len, d) — text tokens

        # Attention of gen tokens → text tokens
        attn  = torch.softmax(
            (q_gen @ k_txt.transpose(-2, -1)) * (d ** -0.5), dim=-1
        )                            # (B, H, n_tok, txt_len)
        energy = attn.mean(dim=(0, 1))   # (n_tok, txt_len)

        # Accumulate (object-word slice applied in compute_mask)
        self._heatmap = energy if self._heatmap is None else self._heatmap + energy
        self._count  += 1
        self._q = self._k = None

    def compute_mask(
        self,
        obj_token_indices: List[int],
        top_frac: float = 0.25,
    ) -> Optional[torch.Tensor]:
        """
        Returns a (n_tok,) bool mask — True where the object is forming.
        obj_token_indices: T5 token positions of the object word in the prompt.
        top_frac: fraction of gen tokens to mark as object zone.
        """
        if self._heatmap is None or self._count == 0:
            return None
        avg  = self._heatmap / self._count         # (n_tok, txt_len)
        if obj_token_indices:
            obj_idx = torch.tensor(
                [i for i in obj_token_indices if i < avg.shape[-1]],
                dtype=torch.long,
            )
            if len(obj_idx) == 0:
                obj_idx = torch.arange(min(5, avg.shape[-1]))
        else:
            obj_idx = torch.arange(min(5, avg.shape[-1]))

        energy    = avg[:, obj_idx].sum(dim=-1)        # (n_tok,)
        threshold = torch.quantile(energy, 1.0 - top_frac)
        return energy >= threshold                      # (n_tok,) bool

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── K/V injector (with optional spatial mask) ─────────────────────────────────

class _KVInject:
    def __init__(
        self,
        kv_store: Dict[int, Dict[str, torch.Tensor]],
        n_tok:    int,
        alpha_k:  float = 0.85,
        alpha_v:  float = 0.50,
    ):
        self.kv_store    = kv_store
        self.n_tok       = n_tok
        self.alpha_k     = alpha_k
        self.alpha_v     = alpha_v
        self.token_mask: Optional[torch.Tensor] = None   # (n_tok,) bool or None
        self._hooks      = []

    def register(self, pipe):
        for i in _TIER_A_DOUBLE:
            b = pipe.transformer.transformer_blocks[i]
            self._add(b.attn.to_k, i, "k")
            self._add(b.attn.to_v, i, "v")
        for j in _TIER_A_SINGLE:
            b = pipe.transformer.single_transformer_blocks[j]
            self._add(b.attn.to_k, _N_DOUBLE + j, "k")
            self._add(b.attn.to_v, _N_DOUBLE + j, "v")

    def _add(self, module, layer_idx: int, kv_type: str):
        alpha = self.alpha_k if kv_type == "k" else self.alpha_v
        n     = self.n_tok

        def hook(mod, inp, out):
            entry = self.kv_store.get(layer_idx, {})
            src   = entry.get(kv_type)
            if src is None:
                return out
            src = src.to(out.device, out.dtype)

            if self.token_mask is not None:
                # Spatially localized: inject only at object token positions
                # mask_f shape (1, n_tok, 1) for broadcasting
                mask_f = self.token_mask.float().to(out.device).unsqueeze(0).unsqueeze(-1)
            else:
                # Phase 1: global injection (all gen tokens)
                mask_f = torch.ones(1, n, 1, device=out.device, dtype=out.dtype)

            result          = out.clone()
            result[:, :n, :] = (
                (1.0 - alpha * mask_f) * out[:, :n, :]
                + alpha * mask_f * src
            )
            return result

        self._hooks.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Token-finding utility ─────────────────────────────────────────────────────

def _find_obj_token_indices(pipe, prompt: str, obj_name: str) -> List[int]:
    """
    Return T5 token indices of obj_name inside the encoded prompt.
    Falls back to the first 5 non-BOS tokens if not found.
    """
    try:
        tok  = pipe.tokenizer_2
        full = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        obj  = tok(obj_name, add_special_tokens=False).input_ids
        for start in range(len(full) - len(obj) + 1):
            if full[start: start + len(obj)] == obj:
                return list(range(start, start + len(obj)))
    except Exception:
        pass
    return list(range(5))   # safe fallback


# ── Public API ────────────────────────────────────────────────────────────────

@torch.no_grad()
def capture_obj_kv(
    pipe,
    obj_img: Image.Image,
    height:  int,
    width:   int,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], int]:
    """
    Single forward pass on clean obj_img at t=0.5 to capture
    K/V at all 13 TIER_A layers.
    Returns (kv_store, n_tok).
    """
    device = pipe.device
    dtype  = next(pipe.transformer.parameters()).dtype
    vae_sf = pipe.vae_scale_factor
    h_lat  = height // vae_sf
    w_lat  = width  // vae_sf
    n_tok  = (h_lat // 2) * (w_lat // 2)

    obj_z  = _encode_image(pipe, obj_img, height, width, dtype)
    grey   = Image.new("RGB", (width, height), (128, 128, 128))
    ref_z  = _encode_image(pipe, grey,    height, width, dtype)

    combined     = torch.cat([_pack(obj_z), _pack(ref_z)], dim=1)
    single_ids   = _image_ids(h_lat, w_lat, device, dtype)
    combined_ids = torch.cat([single_ids, single_ids], dim=0)

    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt="", prompt_2=None, device=device,
        num_images_per_prompt=1, max_sequence_length=512,
    )
    t_norm   = torch.tensor([0.5], device=device, dtype=dtype)
    guidance = torch.tensor([1.0], device=device, dtype=dtype)

    capturer = _KVCapture(n_tok)
    capturer.register(pipe)
    try:
        pipe.transformer(
            hidden_states=combined, timestep=t_norm, guidance=guidance,
            pooled_projections=pooled_embeds, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=combined_ids, return_dict=False,
        )
    finally:
        capturer.remove()

    print(f"  [KV] Captured at {len(capturer.kv)}/13 TIER_A layers")
    return capturer.kv, n_tok


@torch.no_grad()
def run_kv_guided_insertion(
    pipe,
    scene:      Image.Image,
    prompt:     str,
    obj_name:   str,
    kv_store:   Dict[int, Dict[str, torch.Tensor]],
    n_tok:      int,
    seed:       int,
    num_steps:  int,
    guidance:   float,
    height:     int,
    width:      int,
    alpha_k:    float = 0.85,
    alpha_v:    float = 0.50,
    warm_steps: int   = 18,
    top_frac:   float = 0.25,
) -> Image.Image:
    """
    Kontext denoising with two-phase K/V injection:
      Phase 1 (0→warm_steps): global injection + DAAM accumulation.
      Phase 2 (warm_steps→end): localized injection at DAAM mask only.
    """
    device  = pipe.device
    dtype   = next(pipe.transformer.parameters()).dtype
    vae_sf  = pipe.vae_scale_factor
    h_lat   = height // vae_sf
    w_lat   = width  // vae_sf
    n_heads = pipe.transformer.config.num_attention_heads
    d_head  = pipe.transformer.config.attention_head_dim

    scene_tok    = _pack(_encode_image(pipe, scene, height, width, dtype))
    single_ids   = _image_ids(h_lat, w_lat, device, dtype)
    combined_ids = torch.cat([single_ids, single_ids], dim=0)

    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device,
        num_images_per_prompt=1, max_sequence_length=512,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    latents   = torch.randn(
        (1, 16, h_lat, w_lat), device=device, dtype=dtype, generator=generator,
    )

    mu = _compute_scheduler_mu(pipe, n_tok)
    if mu is not None:
        pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=device)

    guidance_t = torch.tensor([guidance], device=device, dtype=dtype)

    # Find object-word token indices in T5-encoded prompt
    obj_token_indices = _find_obj_token_indices(pipe, prompt, obj_name)
    print(f"  [DAAM] Object '{obj_name}' → T5 token indices: {obj_token_indices}")

    # Phase 1: DAAM-only (no injection) — let the model place the object freely
    daam     = _DAAMAccumulator(n_tok, n_heads, d_head)
    injector = _KVInject(kv_store, n_tok, alpha_k, alpha_v)

    daam.register(pipe)
    # injector NOT registered yet — Phase 1 is observation only
    try:
        for i, t in enumerate(pipe.scheduler.timesteps):

            # Phase transition: build mask, remove DAAM hooks, start injection
            if i == warm_steps:
                mask = daam.compute_mask(obj_token_indices, top_frac)
                daam.remove()
                if mask is not None:
                    injector.token_mask = mask
                    n_obj = int(mask.sum().item())
                    print(f"  [DAAM] Mask locked at step {i}: "
                          f"{n_obj}/{n_tok} tokens ({n_obj/n_tok*100:.1f}%) — starting injection")
                else:
                    print(f"  [DAAM] Mask unavailable — skipping injection")
                injector.register(pipe)

            gen_tok  = _pack(latents)
            combined = torch.cat([gen_tok, scene_tok], dim=1)
            t_norm   = (t.float() / 1000.0).to(dtype).expand(1)

            out = pipe.transformer(
                hidden_states=combined, timestep=t_norm, guidance=guidance_t,
                pooled_projections=pooled_embeds, encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, img_ids=combined_ids, return_dict=False,
            )[0]

            v_gen   = out[:, :n_tok, :]
            v_gen   = _unpack(v_gen, height, width, vae_sf)
            latents = pipe.scheduler.step(v_gen, t, latents, return_dict=False)[0]

            if (i + 1) % 5 == 0 or i == 0:
                phase = "DAAM" if i < warm_steps else "KV-inject"
                print(f"    [{phase}] Step {i+1}/{num_steps}")

    finally:
        injector.remove()
        if daam._hooks:
            daam.remove()

    latents_out = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image_t     = pipe.vae.decode(latents_out, return_dict=False)[0]
    return pipe.image_processor.postprocess(image_t, output_type="pil")[0]
