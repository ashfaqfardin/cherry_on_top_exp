from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image

from .mask_ops import _neutralize_white_bg

_TIER_A         = {0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56}
_SINGLE_TXT_LEN = 512


# ── Spatial alignment ─────────────────────────────────────────────────────────

def _spatial_align_k_from_obj(
    k_obj: torch.Tensor,      # (1, H, n_gen, d) — K from obj_img capture pass
    obj_mask: np.ndarray,     # (n_gen,) bool — non-background tokens in obj_img
    target_zone: np.ndarray,  # (n_gen,) bool — target zone in scene
    h_lat: int,
    w_lat: int,
) -> torch.Tensor:
    """
    Returns (1, H, n_gen, d): for each scene position, the K from the
    geometrically corresponding position in obj_img (bbox-to-bbox mapping).

    Fixes the mean-pool collapse: instead of every target token seeing the
    same average K, handlebar area gets handlebar K, wheel area gets wheel K.
    """
    obj_2d = obj_mask.reshape(h_lat, w_lat)
    tgt_2d = target_zone.reshape(h_lat, w_lat)

    def _bbox(m):
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return 0, h_lat - 1, 0, w_lat - 1
        return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])

    r0_o, r1_o, c0_o, c1_o = _bbox(obj_2d)
    r0_t, r1_t, c0_t, c1_t = _bbox(tgt_2d)

    RR, CC = np.meshgrid(np.arange(h_lat, dtype=np.float32),
                         np.arange(w_lat, dtype=np.float32), indexing='ij')

    def _map(pos, p0t, p1t, p0o, p1o):
        if p1t == p0t:
            return np.full_like(pos, (p0o + p1o) * 0.5)
        return np.clip(p0o + (pos - p0t) / float(p1t - p0t) * (p1o - p0o), p0o, p1o)

    R_idx = np.clip(np.round(_map(RR, r0_t, r1_t, r0_o, r1_o)).astype(int), 0, h_lat - 1)
    C_idx = np.clip(np.round(_map(CC, c0_t, c1_t, c0_o, c1_o)).astype(int), 0, w_lat - 1)
    flat  = torch.from_numpy((R_idx * w_lat + C_idx).reshape(-1)).to(k_obj.device)
    return k_obj[:, :, flat, :]


# ── K/V capture ──────────────────────────────────────────────────────────────

class _KVCapture:
    def __init__(self, n_gen: int):
        self.n_gen = n_gen
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2); k = torch.cat([ek, k], dim=2); v = torch.cat([ev, v], dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb); k = apply_rotary_emb(k, image_rotary_emb)
        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                self.kv[self._layer] = (k[:,:,r_lo:r_hi,:].detach().cpu(),
                                        v[:,:,r_lo:r_hi,:].detach().cpu())
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


# ── _KVInject: simple obj K/V injection into target zone ─────────────────────

class _KVInject:
    def __init__(self, n_gen, obj_kv, obj_mask_np, target_zone, obj_strength, n_steps, cutoff_frac):
        self.n_gen = n_gen; self.obj_kv = obj_kv; self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone; self.obj_strength = obj_strength
        self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2); k = k.view(B,-1,attn.heads,hd).transpose(1,2); v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        lo = int(self.cutoff_frac[0]*self.n_steps); hi = int(self.cutoff_frac[1]*self.n_steps)
        if lo <= self._step < hi and self._layer in self.obj_kv:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off, img_off + self.n_gen
            k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
            k_r, v_r = self.obj_kv[self._layer]
            k_r = k_r.to(k.device, k.dtype); v_r = v_r.to(v.device, v.dtype)
            obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
            if obj_idx.numel() > 0:
                k_obj = k_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(k_gen)
                v_obj = v_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(v_gen)
            else:
                k_obj = k_r.mean(dim=2,keepdim=True).expand_as(k_gen)
                v_obj = v_r.mean(dim=2,keepdim=True).expand_as(v_gen)
            tgt = torch.from_numpy(self.target_zone).to(k.device).view(1,1,-1,1)
            s = self.obj_strength
            k_gen = torch.where(tgt,(1-s)*k_gen+s*k_obj,k_gen)
            v_gen = torch.where(tgt,(1-s)*v_gen+s*v_obj,v_gen)
            k = k.clone(); v = v.clone()
            k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


# ── _DualKVInject: background freeze + spatially-aligned obj injection ────────

class _DualKVInject:
    """
    Dual K/V injection for no-collage object insertion.

    Background tokens: scene context K/V at ALL TIER_A steps (KV-Edit-style preservation).
    Object zone tokens: spatially-aligned obj_img K/V during early steps (cutoff_frac gated).
    """
    def __init__(self, n_gen, obj_kv, obj_mask_np, target_zone,
                 obj_strength, bg_strength, n_steps, cutoff_frac,
                 h_lat: int = 0, w_lat: int = 0):
        self.n_gen = n_gen; self.obj_kv = obj_kv; self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone
        self.obj_strength = obj_strength; self.bg_strength = bg_strength
        self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        self.h_lat = h_lat; self.w_lat = w_lat
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2)
        k = k.view(B,-1,attn.heads,hd).transpose(1,2)
        v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off,              img_off + self.n_gen
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
                tgt = torch.from_numpy(self.target_zone).to(k.device)
                if self.bg_strength > 0:
                    k_ref = k[:,:,r_lo:r_hi,:]; v_ref = v[:,:,r_lo:r_hi,:]
                    bg_t  = (~tgt).view(1,1,-1,1); s_bg = self.bg_strength
                    k_gen = torch.where(bg_t, (1-s_bg)*k_gen + s_bg*k_ref, k_gen)
                    v_gen = torch.where(bg_t, (1-s_bg)*v_gen + s_bg*v_ref, v_gen)
                lo = int(self.cutoff_frac[0]*self.n_steps)
                hi = int(self.cutoff_frac[1]*self.n_steps)
                if lo <= self._step < hi and self._layer in self.obj_kv and self.obj_strength > 0:
                    k_r, v_r = self.obj_kv[self._layer]
                    k_r = k_r.to(k.device, k.dtype); v_r = v_r.to(v.device, v.dtype)
                    if self.h_lat > 0 and self.w_lat > 0:
                        k_obj = _spatial_align_k_from_obj(k_r, self.obj_mask_np,
                                                          self.target_zone, self.h_lat, self.w_lat)
                        obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
                        v_obj = (_spatial_align_k_from_obj(v_r, self.obj_mask_np,
                                                           self.target_zone, self.h_lat, self.w_lat)
                                 if obj_idx.numel() > 0
                                 else v_r.mean(dim=2, keepdim=True).expand_as(v_gen))
                    else:
                        obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
                        if obj_idx.numel() > 0:
                            k_obj = k_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(k_gen)
                            v_obj = v_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(v_gen)
                        else:
                            k_obj = k_r.mean(dim=2,keepdim=True).expand_as(k_gen)
                            v_obj = v_r.mean(dim=2,keepdim=True).expand_as(v_gen)
                    tgt_t = tgt.view(1,1,-1,1); s_obj = self.obj_strength
                    k_gen = torch.where(tgt_t, (1-s_obj)*k_gen + s_obj*k_obj, k_gen)
                    v_gen = torch.where(tgt_t, (1-s_obj)*v_gen + s_obj*v_obj, v_gen)
                k = k.clone(); v = v.clone()
                k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


# ── _CollageKVInject: soft-weight collage reference injection ─────────────────

class _CollageKVInject:
    def __init__(self, n_gen, target_zone, obj_strength, n_steps, cutoff_frac,
                 target_weights: Optional[np.ndarray] = None):
        self.n_gen = n_gen; self.target_zone = target_zone
        self.obj_strength = obj_strength; self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        # target_weights: float (n_gen,) in [0,1] from DAAM heatmap.
        # When provided, replaces binary target_zone with a gradient so periphery
        # tokens blend gently instead of snapping at a hard rectangular boundary.
        self.target_weights = target_weights
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2); k = k.view(B,-1,attn.heads,hd).transpose(1,2); v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        lo = int(self.cutoff_frac[0]*self.n_steps); hi = int(self.cutoff_frac[1]*self.n_steps)
        if lo <= self._step < hi and self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off,              img_off + self.n_gen
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
                k_ref = k[:,:,r_lo:r_hi,:];         v_ref = v[:,:,r_lo:r_hi,:]
                if self.target_weights is not None:
                    w = torch.from_numpy(self.target_weights).to(k.device, k.dtype)
                    s_tok = (self.obj_strength * w).view(1, 1, -1, 1)
                    k_gen = k_gen + s_tok * (k_ref - k_gen)
                    v_gen = v_gen + s_tok * (v_ref - v_gen)
                else:
                    tgt = torch.from_numpy(self.target_zone).to(k.device).view(1,1,-1,1)
                    s = self.obj_strength
                    k_gen = torch.where(tgt,(1.0-s)*k_gen+s*k_ref,k_gen)
                    v_gen = torch.where(tgt,(1.0-s)*v_gen+s*v_ref,v_gen)
                k = k.clone(); v = v.clone()
                k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


# ── Run helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_with_dual_kv_injection(
    pipe, scene, prompt, obj_img, obj_mask_np, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.5, bg_strength=0.8, cutoff_frac=(0.0, 0.6),
) -> Image.Image:
    """No-collage object insertion: background freeze + spatially-aligned obj K/V."""
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    h_lat  = height // (vae_sf * 2)
    w_lat  = width  // (vae_sf * 2)
    n_gen  = h_lat * w_lat
    orig_procs   = pipe.transformer.attn_processors
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=_neutralize_white_bg(obj_img),
         prompt="photorealistic object, studio lighting",
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen0, output_type="latent")
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")
    inject_proc = _DualKVInject(
        n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
        target_zone=target_zone,
        obj_strength=obj_strength, bg_strength=bg_strength,
        n_steps=num_steps, cutoff_frac=cutoff_frac,
        h_lat=h_lat, w_lat=w_lat,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return ck

    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=scene, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, max_sequence_length=512,
        generator=gen1, output_type="pil",
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


@torch.no_grad()
def run_with_kv_injection(
    pipe, canvas, prompt, obj_img, obj_mask_np, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.7, cutoff_frac=(0.0, 0.6), bcg_callback=None,
) -> Image.Image:
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    orig_procs   = pipe.transformer.attn_processors
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=_neutralize_white_bg(obj_img),
         prompt="photorealistic object, studio lighting",
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen0, output_type="latent")
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")
    inject_proc = _KVInject(n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
                             target_zone=target_zone, obj_strength=obj_strength,
                             n_steps=num_steps, cutoff_frac=cutoff_frac)
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(image=canvas, prompt=prompt,
                  num_inference_steps=num_steps, guidance_scale=guidance,
                  height=height, width=width, max_sequence_length=512,
                  generator=gen1, output_type="pil",
                  callback_on_step_end=_step_cb,
                  callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else []).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


@torch.no_grad()
def run_with_collage_kv_injection(
    pipe, canvas, prompt, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.5, cutoff_frac=(0.0, 0.6), bcg_callback=None,
    target_weights: Optional[np.ndarray] = None,
) -> Image.Image:
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    orig_procs  = pipe.transformer.attn_processors
    inject_proc = _CollageKVInject(n_gen=n_gen, target_zone=target_zone,
                                   obj_strength=obj_strength, n_steps=num_steps,
                                   cutoff_frac=cutoff_frac, target_weights=target_weights)
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(image=canvas, prompt=prompt,
                  num_inference_steps=num_steps, guidance_scale=guidance,
                  height=height, width=width, max_sequence_length=512,
                  generator=gen, output_type="pil",
                  callback_on_step_end=_step_cb,
                  callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else []).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


# ── BLD latent background preservation callback ───────────────────────────────

def _make_bcg_latent_callback(
    scene:        Image.Image,
    mask_np:      np.ndarray,
    pipe,
    height:       int,
    width:        int,
    device:       str,
    dtype:        torch.dtype = torch.bfloat16,
    dilation_lat: int = 3,
):
    """
    BLD: at every denoising step, pixels OUTSIDE the placement mask are
    overwritten with the original scene latents (noised to the current sigma).
    Guarantees pixel-perfect background preservation by construction.
    """
    import cv2 as _cv2
    vae_h = height // 8; vae_w = width // 8

    scene_np = np.array(scene.convert("RGB"), dtype=np.float32) / 255.0
    scene_t  = torch.from_numpy(scene_np).permute(2, 0, 1).unsqueeze(0)
    scene_t  = (scene_t * 2.0 - 1.0).to(dtype)
    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0 = pipe.vae.encode(scene_t.to(enc_dev)).latent_dist.sample()
        z0 = (z0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z0    = z0.to(device, dtype)
    noise = torch.randn(z0.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    mask_lat = np.array(
        Image.fromarray(mask_np).resize((vae_w, vae_h), Image.NEAREST)
    )
    mask_lat = (mask_lat > 127).astype(np.uint8)
    if dilation_lat > 0:
        k = dilation_lat * 2 + 1
        mask_lat = _cv2.dilate(mask_lat, np.ones((k, k), np.uint8), iterations=2)
    mask_soft   = _cv2.GaussianBlur(mask_lat.astype(np.float32), (7, 7), 2.0)
    mask_tensor = torch.from_numpy(mask_soft).to(device, dtype)[None, None]
    print(f"    [BLD] Latent mask {vae_h}×{vae_w}, edit zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape[-2:] != (vae_h, vae_w):
            return callback_kwargs
        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))
        z_bg  = ((1.0-sigma)*z0 + sigma*noise).to(latents.dtype).to(latents.device)
        m     = mask_tensor.expand_as(latents).to(latents.dtype).to(latents.device)
        callback_kwargs["latents"] = latents * m + z_bg * (1.0 - m)
        return callback_kwargs

    return _callback


# ── Null-Referenced Feature Delta ─────────────────────────────────────────────
#
# Novel mechanism (no direct prior in literature):
#
#   Pass A:  pipe(image=grey,     prompt=null_prompt)  →  F_null[TIER_A, ref_tokens]
#   Pass B:  pipe(image=obj_col,  prompt=null_prompt)  →  F_obj[TIER_A, ref_tokens]
#   delta  = F_obj  −  F_null    at each TIER_A layer  (shape: 1 × n_gen × d)
#
# Why grey null baseline cancels entanglement:
#   Grey is spatially uniform → background attention contribution is position-invariant
#   → cancels in subtraction → delta ≈ pure object feature signal
#
# Why no explicit RoPE strip is needed here (unlike K/V):
#   We capture pre-projection hidden_states (before Q/K/V projection).
#   RoPE is applied INSIDE the attention to Q and K, not to hidden_states.
#   So the delta lives in pre-RoPE feature space — inherently position-agnostic.
#
# Injection: add delta to gen-token hidden_states at target zone BEFORE Q/K/V
# projections, so the modification propagates through the full attention + MLP at
# that layer (richer than K/V injection which only touches one attention branch).
#
# Scale adaptation: delta is captured at a 1-step noise level; main generation
# spans 28 steps at varying noise levels (different AdaLN scale factors).
# We normalise the delta direction and rescale to the target hidden-state norm
# so the injection magnitude is consistent regardless of timestep.


class _FeatureDeltaCapture:
    """
    Captures image-token hidden_states at TIER_A double-stream layers.

    In FluxKontextPipeline the image token sequence passed to the double-stream
    layers is [gen_tokens | ref_tokens], each of length n_gen.
    We store the full (1, 2*n_gen, d) slice so the caller can extract whichever
    half it needs (ref = second half = positions n_gen:2*n_gen).
    """

    def __init__(self, n_gen: int):
        self.n_gen = n_gen
        self.features: Dict[int, torch.Tensor] = {}   # layer → (1, ≤2·n_gen, d)
        self._layer = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2)
        k = k.view(B,-1,attn.heads,hd).transpose(1,2)
        v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        # hidden_states = image tokens only in double-stream (text is encoder_hidden_states)
        if is_double and self._layer in _TIER_A:
            self.features[self._layer] = hidden_states.detach().cpu()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


class _FeatureDeltaInject:
    """
    Injects null-referenced feature delta into gen-token residual stream.

    At each TIER_A double-stream layer, BEFORE Q/K/V projections:
        delta_dir = delta[layer] / (||delta[layer]|| + ε)   — unit direction
        scale     = ||hidden_states[tgt, :]||.mean()         — adaptive magnitude
        hidden_states[tgt, :] += obj_strength × scale × delta_dir[tgt, :]

    Spatial alignment: target zone token (r,c) gets delta from the same (r,c)
    ref position — the object was pasted at the same spatial coordinates in the
    null reference pass, so positional correspondence is exact.

    Soft weights (target_weights): per-token scalar in [0,1] from DAAM heatmap
    replaces the binary target_zone mask, giving gradient injection at periphery.
    """

    def __init__(self, n_gen, feature_delta, target_zone,
                 obj_strength, n_steps, cutoff_frac,
                 target_weights: Optional[np.ndarray] = None):
        self.n_gen          = n_gen
        self.feature_delta  = feature_delta
        self.target_zone    = target_zone
        self.obj_strength   = obj_strength
        self.n_steps        = n_steps
        self.cutoff_frac    = cutoff_frac
        self.target_weights = target_weights
        self._layer = 0
        self._step  = 0
        # Diagnostics: one row per step (averaged across TIER_A layers)
        # row = [step, delta_norm, hs_norm, ratio, cosine_sim, inj_frac]
        self._diag: Dict[int, list] = {}   # step → accumulated [delta_norm, hs_norm, cos_sim, n]

    def _record(self, step, d_tgt, hs_tgt):
        # d_tgt, hs_tgt: (1, n_tgt, d) float tensors on any device
        with torch.no_grad():
            d_n  = d_tgt.float().norm(dim=-1).mean().item()
            h_n  = hs_tgt.float().norm(dim=-1).mean().item()
            # cosine similarity between delta and hidden state, averaged across target tokens
            dot  = (d_tgt.float() * hs_tgt.float()).sum(dim=-1)           # (1, n_tgt)
            cos  = (dot / (d_tgt.float().norm(dim=-1) * hs_tgt.float().norm(dim=-1) + 1e-8)).mean().item()
        if step not in self._diag:
            self._diag[step] = [0.0, 0.0, 0.0, 0]
        row = self._diag[step]
        row[0] += d_n; row[1] += h_n; row[2] += cos; row[3] += 1

    def print_diag(self):
        if not self._diag:
            return
        print(f"\n    [FDelta] Injection diagnostics  (obj_strength={self.obj_strength})")
        print(f"    {'step':>4}  {'δ_norm':>8}  {'h_norm':>8}  {'ratio':>8}  "
              f"{'cos_sim':>8}  {'eff_str':>9}")
        print(f"    {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)
        cutoff_step = hi - lo
        for step in sorted(self._diag):
            dn, hn, cs, n = self._diag[step]
            dn /= n; hn /= n; cs /= n
            ratio       = dn / (hn + 1e-8)
            step_in_win = max(0, step - lo)
            t           = step_in_win / max(1, cutoff_step)
            step_scale  = (1.0 - t) ** 0.6
            eff_str     = self.obj_strength * step_scale  # at peak w=1.0
            print(f"    {step:>4}  {dn:>8.3f}  {hn:>8.3f}  {ratio:>8.3f}  "
                  f"{cs:>8.3f}  {eff_str:>9.3f}")
        print()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)

        if is_double and lo <= self._step < hi and self._layer in self.feature_delta:
            delta = self.feature_delta[self._layer].to(hidden_states.device, hidden_states.dtype)

            if self.target_weights is not None:
                w = torch.from_numpy(self.target_weights).to(hidden_states.device, hidden_states.dtype)
            else:
                w = torch.from_numpy(self.target_zone.astype(np.float32)).to(
                    hidden_states.device, hidden_states.dtype)

            tgt_idx = torch.from_numpy(np.where(self.target_zone)[0]).to(hidden_states.device)
            if tgt_idx.numel() > 0 and delta.shape[1] >= self.n_gen:
                hidden_states = hidden_states.clone()

                d_tgt    = delta[:, tgt_idx, :]                        # (1, n_tgt, d)
                hs_tgt   = hidden_states[:, tgt_idx, :].detach()
                d_dir    = d_tgt / (d_tgt.norm(dim=-1, keepdim=True) + 1e-8)
                hs_scale = hs_tgt.norm(dim=-1, keepdim=True)

                self._record(self._step, d_tgt, hs_tgt)

                # Cosine-guided step decay: full strength early (structure formation),
                # tapering as the model self-aligns with the delta direction.
                # Exponent 0.6 keeps strength high for first ~30% of steps then falls fast.
                cutoff_step  = hi - lo
                step_in_win  = max(0, self._step - lo)
                t            = step_in_win / max(1, cutoff_step)
                step_scale   = (1.0 - t) ** 0.6

                w_tgt = w[tgt_idx].view(1, -1, 1)
                hidden_states[:, tgt_idx, :] = (
                    hidden_states[:, tgt_idx, :] +
                    self.obj_strength * step_scale * w_tgt * hs_scale * d_dir
                )

        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2)
        k = k.view(B,-1,attn.heads,hd).transpose(1,2)
        v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


@torch.no_grad()
def run_with_feature_delta_injection(
    pipe,
    scene:         Image.Image,      # current scene (Kontext reference for main pass)
    prompt:        str,
    obj_collage:   Image.Image,      # grey_canvas with object pasted at target location
    target_zone:   np.ndarray,       # (n_gen,) bool
    seed:          int,
    num_steps:     int,
    guidance:      float,
    height:        int,
    width:         int,
    obj_strength:  float = 0.5,
    cutoff_frac:   Tuple[float, float] = (0.0, 0.5),
    bcg_callback                      = None,
    target_weights: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Three-pass feature-delta insertion:

    Pass A  (null baseline):  pipe(image=grey)        → F_null[TIER_A, ref_tokens]
    Pass B  (object on null): pipe(image=obj_collage) → F_obj[TIER_A, ref_tokens]
    delta   = F_obj − F_null  (ref-position half of image token sequence)

    Main pass: run_with_collage_kv_injection replacement using _FeatureDeltaInject,
    which adds the adaptive-scaled delta to gen-token hidden_states BEFORE Q/K/V
    projections at each TIER_A layer.

    Why grey as null baseline:
      Uniform grey → attention from background tokens is position-invariant →
      cancels in delta subtraction → minimum base-scene entanglement.

    Why ref-position delta applies to gen-position tokens:
      In FLUX's joint image sequence [gen | ref], ref and gen tokens share the
      same feature space (same projection weights, same dimensionality).
      The spatial correspondence is exact: ref token (r,c) ↔ gen token (r,c).
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    h_lat  = height // (vae_sf * 2)
    w_lat  = width  // (vae_sf * 2)
    n_gen  = h_lat * w_lat

    grey        = Image.new("RGB", (width, height), (128, 128, 128))
    null_prompt = "uniform grey surface, no objects"
    orig_procs  = pipe.transformer.attn_processors

    # ── Pass A: null baseline ─────────────────────────────────────────────────
    cap_null = _FeatureDeltaCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(cap_null)
    gen_a = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=grey, prompt=null_prompt,
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen_a, output_type="latent")

    # ── Pass B: object on null ────────────────────────────────────────────────
    cap_obj = _FeatureDeltaCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(cap_obj)
    gen_b = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=obj_collage, prompt=null_prompt,
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen_b, output_type="latent")

    # ── Compute delta from reference-half of image token sequence ─────────────
    feature_delta: Dict[int, torch.Tensor] = {}
    for layer in cap_null.features:
        if layer not in cap_obj.features:
            continue
        f_null = cap_null.features[layer]   # (1, n_img, d)
        f_obj  = cap_obj.features[layer]
        n_img  = f_null.shape[1]
        if n_img >= 2 * n_gen:
            # Kontext: image tokens = [gen | ref], ref is second half
            f_null_ref = f_null[:, n_gen : 2 * n_gen, :]
            f_obj_ref  = f_obj[:,  n_gen : 2 * n_gen, :]
        else:
            # Single-image pass (fallback): use all tokens
            f_null_ref = f_null
            f_obj_ref  = f_obj
        feature_delta[layer] = f_obj_ref - f_null_ref   # (1, n_gen, d)

    print(f"    [FDelta] Delta captured at {len(feature_delta)}/{len(_TIER_A)} TIER_A layers")
    if feature_delta:
        sample = next(iter(feature_delta.values()))
        d_norms = sample.norm(dim=-1).squeeze()
        print(f"    [FDelta] ||delta|| min={d_norms.min():.3f} "
              f"mean={d_norms.mean():.3f} max={d_norms.max():.3f}")

    # ── Main generation with delta injection ──────────────────────────────────
    inject = _FeatureDeltaInject(
        n_gen=n_gen, feature_delta=feature_delta, target_zone=target_zone,
        obj_strength=obj_strength, n_steps=num_steps, cutoff_frac=cutoff_frac,
        target_weights=target_weights,
    )
    pipe.transformer.set_attn_processor(inject)

    def _step_cb(pr, si, ts, ck):
        inject._step = si + 1; inject._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen_main = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=scene, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, max_sequence_length=512,
        generator=gen_main, output_type="pil",
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else [],
    ).images[0]

    pipe.transformer.set_attn_processor(orig_procs)
    inject.print_diag()
    return result
