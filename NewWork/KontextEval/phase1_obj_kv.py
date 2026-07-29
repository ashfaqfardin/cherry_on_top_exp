# -*- coding: utf-8 -*-
"""
phase1_obj_kv.py — Object-Identity-Preserving Insertion via K/V Injection

Problem with phase1_sketch_vlm.py
----------------------------------
  VLM writes a text description of the object → Kontext generates its own
  version from that text → appearance diverges from the sketch-generated obj.
  Text cannot encode exact visual identity (color nuance, frame shape, etc.).

Solution
--------
  Capture K/V from the actual obj_img at TIER_A (content-similarity) layers
  via a single forward pass, then replay them into the denoising loop at the
  spatial tokens where the new object appears.

  At TIER_A layers, these K/V encode the object's appearance at the semantic
  feature level.  Injecting them into the generation pulls that region toward
  the captured appearance without forcing a pixel-perfect paste (non-rigid).

Pipeline per object
-------------------
  Stage A  : Sketch → LoRA → obj_img          (same as sketch_vlm)
  Stage VLM: VLM(scene, obj_img) → placement_prompt  (same as sketch_vlm)
  Stage K  : Single Kontext forward pass with obj_img as canvas
             → capture K/V at TIER_A layers (ref-token slice)
  Stage B  : Full denoising (28 steps) with:
               scene as canvas  → provides room background context
               Steps 0-7        → DAAM saliency accumulates WHERE object lands
               Step 7           → target mask derived from saliency
               Steps 7-28:
                 background zone  → scene ref K/V injection (room locked)
                 target zone      → mean-pooled obj K/V injection (identity locked)

Usage
-----
  python NewWork/KontextEval/phase1_obj_kv.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_obj_kv \\
      --vlm_model Qwen/Qwen2-VL-2B-Instruct

Key flags
---------
  --obj_strength     float  Strength of obj K/V injection into target zone. (default 0.7)
                            Higher → more appearance fidelity, less natural integration.
  --inject_strength  float  Background K/V freeze strength. (default 0.9)
  --top_k_frac       float  Fraction of gen tokens used as object region. (default 0.10)
  --derive_step      int    Step at which DAAM mask is computed. (default 7)
  --cutoff_frac      str    "start,end" injection active window. (default "0.0,0.6")
  --capture_steps    int    Steps for K/V capture pass. (default 1, keep fast)
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

# ── inject framework from IncrementalEdit ────────────────────────────────────
_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)

from kontext_injection import (  # noqa: E402
    TIER_A, N_DOUBLE, N_LAYERS,
    InjectionState, ZoneMasks, KontextInjectionProcessor,
    get_t5_token_indices, set_determinism,
)
from mask_ops import topk_mask  # noqa: E402


def _step_active(step: int, n_steps: int, frac: Tuple[float, float]) -> bool:
    return int(frac[0] * n_steps) <= step < int(frac[1] * n_steps)


# ── import utilities from sibling files ──────────────────────────────────────
_comp_path = str(Path(__file__).parent / "phase1_composite.py")
_cspec = importlib.util.spec_from_file_location("phase1_composite", _comp_path)
_cmod = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
save_grid    = _cmod.save_grid
run_standard = _cmod.run_standard

_vlm_path = str(Path(__file__).parent / "phase1_sketch_vlm.py")
_vspec = importlib.util.spec_from_file_location("phase1_sketch_vlm", _vlm_path)
_vmod = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vmod)
generate_from_sketch      = _vmod.generate_from_sketch
vlm_generate_kontext_prompt = _vmod.vlm_generate_kontext_prompt

_sk_path = str(Path(__file__).parent / "phase1_sketch.py")
_sspec = importlib.util.spec_from_file_location("phase1_sketch", _sk_path)
_smod = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_smod)
load_vlm = _smod.load_vlm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "white ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT  = "A modern living room with a sofa and a wooden coffee table."
LORA_ID      = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."
_SEP = "═" * 60


# ── Background neutralization ─────────────────────────────────────────────────

def _neutralize_white_bg(img: Image.Image, threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    arr[white] = fill
    return Image.fromarray(arr)


# ── Stage K: capture K/V from obj_img ─────────────────────────────────────────

class _ObjKVCapture:
    """
    Minimal attention processor that performs standard attention AND captures
    K/V from the Kontext reference-token slice at TIER_A layers.

    In Kontext, hidden_states = [gen_tokens | ref_tokens].  The ref-token
    slice starts at img_offset + n_gen.  We store these (K, V) on CPU for
    later injection.

    One instance is used for exactly one forward call sequence (one capture
    pass).  Call reset() before re-use.
    """

    def __init__(self, n_gen: int, tier_a: Set[int], single_txt_len: int = 512):
        self.n_gen = n_gen
        self.tier_a = tier_a
        self.single_txt_len = single_txt_len
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def reset(self):
        self._layer = 0
        self.kv = {}

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        inner_dim = k.shape[-1]
        head_dim = inner_dim // attn.heads

        q = q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # Capture K/V from ref-token slice at TIER_A layers
        if self._layer in self.tier_a:
            img_offset = txt_len if is_double else self.single_txt_len
            ref_lo = img_offset + self.n_gen
            ref_hi = ref_lo + self.n_gen  # n_ref == n_gen (same resolution canvas)
            if k.shape[2] >= ref_hi:
                self.kv[self._layer] = (
                    k[:, :, ref_lo:ref_hi, :].detach().cpu(),
                    v[:, :, ref_lo:ref_hi, :].detach().cpu(),
                )

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                              attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            self._layer += 1
            return out, enc_out

        self._layer += 1
        return out


@torch.no_grad()
def capture_obj_kv(
    pipe,
    obj_img: Image.Image,
    height: int,
    width: int,
    device: str,
    vital_layers: List[int],
    seed: int = 0,
    capture_steps: int = 1,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Run a minimal Kontext denoising pass with obj_img as the canvas.
    Captures K/V from the reference-token slice at TIER_A layers.

    At t≈1 (the first step), the transformer encodes the reference (obj_img)
    very strongly in the K/V — the noisy gen latents have not yet established
    any structure, so the reference dominates.  One step is enough.

    Returns: {layer_idx: (K_cpu, V_cpu)} for each TIER_A layer captured.
    """
    vae_sf  = getattr(pipe, "vae_scale_factor", 8)
    patch   = getattr(pipe.transformer.config, "patch_size", 2)
    n_gen   = (height // (vae_sf * patch)) * (width // (vae_sf * patch))

    obj_neutral  = _neutralize_white_bg(obj_img)
    capture_proc = _ObjKVCapture(n_gen=n_gen, tier_a=set(vital_layers), single_txt_len=512)
    pipe.transformer.set_attn_processor(capture_proc)

    generator = set_determinism(seed)
    pipe(
        image=obj_neutral,
        prompt="photorealistic object, studio lighting",
        num_inference_steps=capture_steps,
        guidance_scale=1.0,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
        output_type="latent",
    )

    captured = dict(capture_proc.kv)
    print(f"    [K] Captured K/V at {len(captured)}/{len(vital_layers)} TIER_A layers.")
    return captured


# ── Injection processor ───────────────────────────────────────────────────────

class ObjKVInject(KontextInjectionProcessor):
    """
    K/V injection processor with dual-source injection:

      background zone  →  scene ref K/V  (locks room background; same as
                          KontextInjectionProcessor's standard behaviour)
      shell zone       →  scene ref K only (silhouette edge)
      target zone      →  mean-pooled obj K/V  [NEW — locks object appearance]

    In "reasoning" mode (steps 0 → derive_step), DAAM saliency is accumulated
    to discover the object placement area — same as the parent class.
    In "edit" mode (steps derive_step → end), both background freeze AND
    obj appearance injection are active.

    obj K/V are mean-pooled over their spatial dimension before injection.
    This handles the spatial mismatch between obj_img (object centered) and
    the target zone in the scene (object at any location) while still
    transmitting the global appearance signal (color, texture, style) that
    TIER_A layers carry.
    """

    def __init__(self, state: InjectionState,
                 obj_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                 obj_strength: float = 0.7):
        super().__init__(state, total_layers=N_LAYERS)
        self.obj_kv      = obj_kv
        self.obj_strength = obj_strength

    def _inject(self, k, v, img_offset: int):
        st    = self.state
        layer = self.cur_layer

        if not _step_active(st.cur_step, st.n_steps, st.cutoff_frac):
            return k, v

        n_gen = st.n_gen
        gen_lo, gen_hi = img_offset, img_offset + n_gen
        ref_lo, ref_hi = gen_hi, gen_hi + st.n_ref

        k_ref = k[:, :, ref_lo:ref_hi, :]   # scene reference K
        v_ref = v[:, :, ref_lo:ref_hi, :]   # scene reference V
        k_gen = k[:, :, gen_lo:gen_hi, :].clone()
        v_gen = v[:, :, gen_lo:gen_hi, :].clone()

        zones = st.zones
        s_bg  = st.strength

        # ── Background: freeze to scene reference (preserves room) ────────────
        if zones is not None and zones.background.any():
            bg    = zones.background.view(1, 1, -1, 1)
            k_gen = torch.where(bg, (1 - s_bg) * k_gen + s_bg * k_ref, k_gen)
            v_gen = torch.where(bg, (1 - s_bg) * v_gen + s_bg * v_ref, v_gen)

        # ── Shell: K only freeze (silhouette edge) ────────────────────────────
        if zones is not None and zones.shell.any():
            sh    = zones.shell.view(1, 1, -1, 1)
            k_gen = torch.where(sh, (1 - s_bg) * k_gen + s_bg * k_ref, k_gen)

        # ── Target: inject mean-pooled obj K/V (object appearance lock) ───────
        if (zones is not None and zones.target.any() and layer in self.obj_kv):
            k_obj_raw, v_obj_raw = self.obj_kv[layer]
            k_obj_raw = k_obj_raw.to(k.device, k.dtype)
            v_obj_raw = v_obj_raw.to(v.device, v.dtype)

            # Mean-pool over spatial positions so every target token gets
            # the same global appearance summary — robust to the mismatch
            # between obj_img's spatial layout and the target zone's location.
            k_obj = k_obj_raw.mean(dim=2, keepdim=True).expand_as(k_gen)
            v_obj = v_obj_raw.mean(dim=2, keepdim=True).expand_as(v_gen)

            tgt   = zones.target.view(1, 1, -1, 1)
            s_obj = self.obj_strength
            k_gen = torch.where(tgt, (1 - s_obj) * k_gen + s_obj * k_obj, k_gen)
            v_gen = torch.where(tgt, (1 - s_obj) * v_gen + s_obj * v_obj, v_gen)

        k = k.clone()
        v = v.clone()
        k[:, :, gen_lo:gen_hi, :] = k_gen
        v[:, :, gen_lo:gen_hi, :] = v_gen
        return k, v


# ── Stage B: denoising with DAAM placement + obj K/V injection ───────────────

@torch.no_grad()
def run_obj_appearance_edit(
    pipe,
    scene:         Image.Image,
    prompt:        str,
    obj_kv:        Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    obj_noun:      str,
    vital_layers:  List[int],
    height:        int,
    width:         int,
    seed:          int   = 42,
    num_steps:     int   = 28,
    guidance:      float = 2.5,
    inject_strength: float = 0.9,
    obj_strength:  float = 0.7,
    cutoff_frac:   Tuple[float, float] = (0.0, 0.6),
    derive_step:   int   = 7,
    top_k_frac:    float = 0.10,
    device:        str   = "cuda",
) -> Image.Image:
    """
    Edit scene with appearance-preserving K/V injection.

    Steps 0 → derive_step : DAAM saliency accumulates (reasoning mode).
                             Background frozen everywhere (nothing changes yet).
    Step  derive_step      : Target mask derived from saliency.
    Steps derive_step → end: Edit mode:
                               background → scene ref K/V freeze
                               target     → obj K/V injection
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    patch  = getattr(pipe.transformer.config, "patch_size", 2)
    h_lat  = height // (vae_sf * patch)
    w_lat  = width  // (vae_sf * patch)
    n_gen  = h_lat * w_lat

    everywhere = np.ones(n_gen, dtype=bool)
    nothing    = np.zeros(n_gen, dtype=bool)

    concept_idx = get_t5_token_indices(pipe, prompt, obj_noun)

    if not concept_idx:
        # Fallback: noun not found → run with background freeze only (no DAAM, no target)
        print(f"    [!] '{obj_noun}' not in T5 tokens — background-freeze only (no obj injection).")
        state = InjectionState(
            mode="edit",
            vital_layers=set(vital_layers),
            n_gen=n_gen, n_ref=n_gen,
            cutoff_frac=cutoff_frac,
            strength=inject_strength,
            n_steps=num_steps,
        )
        state.zones = ZoneMasks(
            background=everywhere, shell=nothing, target=nothing,
        ).to_device(device)
        proc = ObjKVInject(state, obj_kv=obj_kv, obj_strength=0.0)
        proc._single_txt_len = 512
        pipe.transformer.set_attn_processor(proc)
        generator = set_determinism(seed)
        return pipe(
            image=scene, prompt=prompt, num_inference_steps=num_steps,
            guidance_scale=guidance, height=height, width=width,
            max_sequence_length=512, generator=generator, output_type="pil",
        ).images[0]

    # Reasoning state — background fully frozen so DAAM reflects WHERE the
    # model would place the object, not where it drifts due to edits.
    state = InjectionState(
        mode="reasoning",
        vital_layers=set(vital_layers),
        n_gen=n_gen, n_ref=n_gen,
        cutoff_frac=cutoff_frac,
        strength=inject_strength,
        n_steps=num_steps,
        derive_step=derive_step,
        concept_token_idx=concept_idx,
    )
    state.concept_layers = (
        {l for l in vital_layers if l < N_DOUBLE} or set(range(N_DOUBLE))
    )
    state.zones = ZoneMasks(
        background=everywhere, shell=nothing, target=nothing,
    ).to_device(device)

    proc = ObjKVInject(state, obj_kv=obj_kv, obj_strength=obj_strength)
    proc._single_txt_len = 512
    pipe.transformer.set_attn_processor(proc)

    def _callback(pipe_ref, step_index, timestep, callback_kwargs):
        if state.mode == "reasoning" and step_index == state.derive_step:
            if state.saliency_accum is not None and state.saliency_captures > 0:
                sal    = (state.saliency_accum / state.saliency_captures).cpu().numpy()
                target = topk_mask(sal, top_k_frac)
                bg     = np.logical_and(everywhere, ~target)
                state.zones = ZoneMasks(
                    background=bg, shell=nothing, target=target,
                ).to_device(device)
                pct = 100.0 * target.mean()
                print(f"      [DAAM@{step_index}] object region: "
                      f"{target.sum()} / {n_gen} tokens ({pct:.1f}%)")
            else:
                print(f"      [DAAM@{step_index}] no saliency captured — staying frozen.")
            state.mode = "edit"
        return callback_kwargs

    generator = set_determinism(seed)
    result = pipe(
        image=scene,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
        output_type="pil",
        callback_on_step_end=_callback,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]

    return result


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_obj_kv_chain(
    pipe,
    base:          Image.Image,
    edits:         List[dict],
    sketch_dir:    str,
    lora_id:       str,
    vlm_pair:      Tuple,
    vital_layers:  List[int],
    seed:          int,
    num_steps:     int,
    lora_guidance: float,
    scene_guidance:float,
    inject_strength: float,
    obj_strength:  float,
    cutoff_frac:   Tuple[float, float],
    derive_step:   int,
    top_k_frac:    float,
    capture_steps: int,
    height:        int,
    width:         int,
    out_dir:       str,
    device:        str,
) -> List[Image.Image]:
    """
    Incremental insertion: Base → +Bike → +Vase → +Ball.

    Per object:
      Stage A  : sketch → LoRA → obj_img
      Stage VLM: VLM(scene, obj_img) → placement_prompt
      Stage K  : capture_obj_kv(obj_img) → captured K/V at TIER_A
      Stage B  : run_obj_appearance_edit(scene, placement_prompt, captured_kv)
    """
    vlm_model, vlm_proc = vlm_pair
    results = [base]
    scene   = base

    for i, edit in enumerate(edits):
        name = edit["name"]
        desc = edit["description"]
        sketch_path = os.path.join(sketch_dir, f"{name}.png")

        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}\n"
                f"Save your sketch as '{name}.png' in --sketch_dir."
            )

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}")
        print(f"{'─'*60}")

        # Stage A: sketch → object image
        print(f"  [A] Generating '{desc}' from sketch ...")
        obj_img = generate_from_sketch(
            pipe=pipe, sketch_path=sketch_path, description=desc,
            seed=seed, num_steps=num_steps, guidance=lora_guidance,
            height=height, width=width, lora_id=lora_id, device=device,
        )
        obj_path = os.path.join(out_dir, f"obj_gen_{name}.png")
        obj_img.save(obj_path)
        print(f"      Saved: {obj_path}")

        # Stage VLM: scene + obj_img → Kontext placement prompt
        print(f"  [VLM] Generating placement prompt for '{name}' ...")
        kontext_prompt = vlm_generate_kontext_prompt(
            vlm_model=vlm_model, vlm_processor=vlm_proc,
            scene_img=scene, obj_img=obj_img, description=desc,
        )
        print(f"\n  {_SEP}")
        print(f"  [VLM → Kontext prompt]  {name}")
        print(f"  {_SEP}")
        for line in kontext_prompt.splitlines():
            print(f"  {line}")
        print(f"  {_SEP}\n")
        prompt_path = os.path.join(out_dir, f"vlm_prompt_{name}.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(kontext_prompt)
        print(f"      Saved: {prompt_path}")

        # Stage K: capture obj K/V
        print(f"  [K] Capturing obj K/V from {name} image ({capture_steps}-step pass) ...")
        obj_kv = capture_obj_kv(
            pipe=pipe, obj_img=obj_img, height=height, width=width,
            device=device, vital_layers=vital_layers, seed=seed,
            capture_steps=capture_steps,
        )

        # Stage B: denoising with DAAM placement + obj injection
        # Noun = last word before any "with ..." clause, so "vase with flowers" → "vase"
        obj_noun = desc.split(" with ")[0].split()[-1].lower()
        print(f"  [B] Editing scene  (obj_noun='{obj_noun}', obj_strength={obj_strength}) ...")
        next_scene = run_obj_appearance_edit(
            pipe=pipe, scene=scene, prompt=kontext_prompt, obj_kv=obj_kv,
            obj_noun=obj_noun, vital_layers=vital_layers,
            height=height, width=width, seed=seed, num_steps=num_steps,
            guidance=scene_guidance, inject_strength=inject_strength,
            obj_strength=obj_strength, cutoff_frac=cutoff_frac,
            derive_step=derive_step, top_k_frac=top_k_frac, device=device,
        )
        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        scene = next_scene
        results.append(scene)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Appearance-preserving sketch-to-scene insertion via K/V injection."
    )
    p.add_argument("--sketch_dir",       required=True,
                   help="Folder containing {name}.png hand-drawn sketches.")
    p.add_argument("--hf_token",         required=True)
    p.add_argument("--cache_dir",        default="./models")
    p.add_argument("--out_dir",          default="results/phase1_obj_kv")
    p.add_argument("--config",           default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",          default=LORA_ID)
    p.add_argument("--lora_guidance",    type=float, default=4.0,
                   help="guidance_scale for Stage A LoRA generation.")
    p.add_argument("--guidance",         type=float, default=2.5,
                   help="guidance_scale for Stage B scene editing.")
    p.add_argument("--inject_strength",  type=float, default=0.9,
                   help="Background freeze strength (scene ref K/V injection).")
    p.add_argument("--obj_strength",     type=float, default=0.7,
                   help="Object appearance lock strength (obj K/V injection). "
                        "Higher → more fidelity to sketch; lower → more natural integration.")
    p.add_argument("--cutoff_frac",      default="0.0,0.6",
                   help="Injection active window as 'start,end' fractions of num_steps.")
    p.add_argument("--derive_step",      type=int, default=7,
                   help="Step at which DAAM object-placement mask is derived.")
    p.add_argument("--top_k_frac",       type=float, default=0.10,
                   help="Fraction of gen tokens identified as object region (DAAM).")
    p.add_argument("--capture_steps",    type=int, default=1,
                   help="Number of steps for K/V capture pass (1 is sufficient).")
    p.add_argument("--vital_layers",     default="tier_a",
                   help="Layers to inject. 'tier_a' (default), 'all', or comma list.")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_steps",        type=int,   default=28)
    p.add_argument("--height",           type=int,   default=1024)
    p.add_argument("--width",            type=int,   default=1024)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--vlm_model",        default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vlm_device",       default="cpu")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from kontext_injection import resolve_vital_layers
    vital_layers  = resolve_vital_layers(args.vital_layers)
    cutoff_frac   = tuple(float(x) for x in args.cutoff_frac.split(","))

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)
        print(f"Loaded {len(edits)} edits from {args.config}")

    print(f"\n{_SEP}")
    print(f"  phase1_obj_kv  —  Appearance-Preserving K/V Injection")
    print(f"{_SEP}")
    print(f"  Objects       : {[e['name'] for e in edits]}")
    print(f"  Sketch dir    : {args.sketch_dir}")
    print(f"  VLM           : {args.vlm_model}  [{args.vlm_device}]")
    print(f"  LoRA          : {args.lora_id}")
    print(f"  vital_layers  : {vital_layers}")
    print(f"  obj_strength  : {args.obj_strength}")
    print(f"  inject_strength: {args.inject_strength}")
    print(f"  top_k_frac    : {args.top_k_frac}")
    print(f"  derive_step   : {args.derive_step}")
    print(f"  cutoff_frac   : {cutoff_frac}")
    print(f"  Output        : {args.out_dir}")
    print(f"{_SEP}\n")

    # Load VLM
    print("Loading VLM ...")
    vlm_pair = load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    # Load Kontext
    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # Base scene
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base_path = os.path.join(args.out_dir, "step0_base.png")
    base.save(base_path)
    print(f"  Saved: {base_path}")

    # Incremental insertion
    print("\n=== Incremental insertion ===")
    results = run_obj_kv_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        vlm_pair=vlm_pair, vital_layers=vital_layers,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance, scene_guidance=args.guidance,
        inject_strength=args.inject_strength, obj_strength=args.obj_strength,
        cutoff_frac=cutoff_frac, derive_step=args.derive_step,
        top_k_frac=args.top_k_frac, capture_steps=args.capture_steps,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
    )

    # Save final + grid
    final_path = os.path.join(args.out_dir, "KEY_final.png")
    results[-1].save(final_path)
    print(f"\n  Final scene: {final_path}")

    titles    = ["Base"] + [e["name"] for e in edits]
    grid_path = os.path.join(args.out_dir, "KEY_grid.png")
    save_grid(results, titles, grid_path, ncols=len(results))
    print(f"  Grid       : {grid_path}")

    print(f"\n{_SEP}")
    print("  WHAT TO CHECK")
    print(f"{_SEP}")
    print("""
  obj_gen_{name}.png
    Stage A: sketch → realistic object (same as sketch_vlm).

  vlm_prompt_{name}.txt
    Placement prompt from VLM — also printed to terminal.

  result_step{N}_{name}.png
    Key comparison vs sketch_vlm output:
    → Does the object's COLOR/TEXTURE match obj_gen_{name}.png more closely?
    → Is the placement reasonable (VLM-guided)?
    → Background preserved?

  KEY_grid.png
    Base | +Bike | +Vase | +Ball

  Tuning
    --obj_strength 0.4   lower → more natural integration, less identity
    --obj_strength 0.9   higher → stronger identity lock, may look pasted
    --top_k_frac 0.15    wider object region (useful for large objects)
    --top_k_frac 0.06    tighter (small objects like ball)
""")


if __name__ == "__main__":
    main()
