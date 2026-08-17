# -*- coding: utf-8 -*-
"""
phase1_ref_qwen.py — No-mask pipeline: native multi-image reference conditioning

Uses Qwen-Image-Edit-2509's built-in multi-image support (image concatenation
training) to pass scene + object reference as separate images — no canvas
stitching, no crop, no bleed artifacts.

Pipeline per insertion step
----------------------------
  1. Stage A  : sketch → obj_img  (Qwen renders the sketch)
  2. Stage K  : Qwen(image=[scene, obj_img], prompt=place_prompt) → 1024×1024

Pipeline per removal step
--------------------------
  1. Qwen(image=scene, prompt=remove_prompt) → next scene

Usage
------
  python NewWork/KontextEval/phase1_ref_qwen.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token   $HF_TOKEN \\
      --cache_dir  ./models \\
      --out_dir    results/phase1_ref_qwen
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import Dict, Tuple

try:
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen as _apply_rope
except ImportError:
    _apply_rope = None   # type: ignore[assignment]


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]

BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)

_SEP = "═" * 60

LIGHTNING_REPO = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_SUBFOLDER = "Qwen-Image-Edit-2509"

# Scheduler config required by the Lightning distilled weights.
# shift=1.0 + use_dynamic_shifting=True replaces the base model's higher shift.
LIGHTNING_SCHEDULER_CFG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


# ── Pipeline loading ───────────────────────────────────────────────────────────

def load_pipeline(
    model: str = "Qwen/Qwen-Image-Edit-2509",
    hf_token: str | None = None,
    cache_dir: str = "./models",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
    lightning: bool = False,
    lightning_steps: int = 8,
) -> QwenImageEditPlusPipeline:
    # Auto-enable CPU offload when GPU VRAM is likely too small.
    # bf16 transformer (20B) + text encoder (7B) needs ~55 GB.
    # Lightning LoRA does NOT reduce the bf16 transformer size — offload still needed.
    if not cpu_offload and device != "cpu" and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if total_gb < 65:
            print(f"  [AUTO] GPU has {total_gb:.0f} GB — enabling cpu_offload "
                  f"(bf16 model needs ~55 GB)")
            cpu_offload = True

    kwargs: dict = dict(torch_dtype=dtype, token=hf_token, cache_dir=cache_dir)
    if lightning:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
            LIGHTNING_SCHEDULER_CFG
        )
        print(f"  [Lightning] Scheduler: FlowMatch dynamic-shift (shift=1.0, log3 range)")

    pipe = QwenImageEditPlusPipeline.from_pretrained(model, **kwargs)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if lightning:
        weight_name = (
            f"Qwen-Image-Edit-2509-Lightning-{lightning_steps}steps-V1.0-bf16.safetensors"
        )
        print(f"  [Lightning] Downloading LoRA: {LIGHTNING_REPO}/{LIGHTNING_SUBFOLDER}/{weight_name}")
        lora_path = hf_hub_download(
            repo_id=LIGHTNING_REPO,
            filename=f"{LIGHTNING_SUBFOLDER}/{weight_name}",
            token=hf_token,
            cache_dir=cache_dir,
        )
        pipe.load_lora_weights(lora_path)
        print(f"  [Lightning] LoRA loaded — {lightning_steps}-step distillation active")

    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Qwen call ─────────────────────────────────────────────────────────────────

def run_qwen(
    pipe,
    image,           # PIL Image or list of PIL Images
    prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    negative_prompt: str = "blurry, distorted, low quality, watermark, text, artifacts",
) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=image, num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        height=height, width=width,
        generator=gen,
    ).images[0]


# ── Stage A: sketch → object image ────────────────────────────────────────────

def _tight_crop(img: Image.Image, pad: int = 32) -> Image.Image:
    """Crop to the bounding box of non-white pixels, add padding, keep square."""
    arr  = np.array(img.convert("RGB"))
    mask = np.any(arr < 240, axis=2)          # non-white pixels
    if not mask.any():
        return img
    rows  = np.where(mask.any(axis=1))[0]
    cols  = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    r0, r1 = max(0, r0 - pad), min(arr.shape[0], r1 + pad)
    c0, c1 = max(0, c0 - pad), min(arr.shape[1], c1 + pad)
    side   = max(r1 - r0, c1 - c0)           # square crop
    cr = img.crop((c0, r0, c0 + side, r0 + side))
    return cr.resize((512, 512), Image.Resampling.LANCZOS)


def generate_object(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
) -> Image.Image:
    sketch = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    prompt = (
        f"Render this hand-drawn sketch as a photorealistic {description}. "
        f"Place it centered on a plain white background. "
        f"Studio lighting, no shadows, no background objects, high detail."
    )
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    obj = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts, background, room, floor",
        image=sketch, num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        height=height, width=width,
        generator=gen,
    ).images[0]
    # Tight-crop to the object so Picture 2 tokens are mostly object, not white bg
    return _tight_crop(obj)


# ── Prompts ────────────────────────────────────────────────────────────────────

def insert_prompt(description: str) -> str:
    # Pipeline prepends "Picture 1: <img>Picture 2: <img>" with no trailing space,
    # so start with a newline to avoid token boundary issues.
    # Picture 1 = scene, Picture 2 = obj_ref (our image list order).
    return (
        f"\nPicture 1 shows a room. Picture 2 shows a {description}. "
        f"Place the {description} from Picture 2 into the room in Picture 1. "
        f"Choose a natural position off-center — near a wall or corner, resting on the wooden floor. "
        f"Add a realistic contact shadow and match the room lighting. "
        f"Do not move or change any other part of the room."
    )


def remove_prompt(description: str) -> str:
    return (
        f"\nRemove the {description} from this room. "
        f"Fill the area with seamless wooden floor and white wall matching the surroundings. "
        f"Do not change anything else in the room."
    )


# ── SKA (Scene K/V Anchoring) ─────────────────────────────────────────────────
# Diagnostic verdict: blocks 52-57 give +0.020 mean gain at σ<0.3.
# Inject scene K/V from step N into step N+1 to anchor background appearance.

SKA_BLOCKS     = list(range(51, 58))   # inclusive: 52,53,54,55,56,57
SKA_SIGMA_GATE = 0.3                    # activate only for σ < this value


class SKAStore:
    """Persistent K/V cache shared across editing steps.

    Two parallel stores:
      kv     / new_kv     — scene_cond frame (background)
      obj_kv / new_obj_kv — obj frame (current object being inserted)
    Both are injected in the next step; the model routes attention by content.
    """
    def __init__(self):
        self.kv:         Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.new_kv:     Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.obj_kv:     Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.new_obj_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.mode:   str  = "normal"   # "normal" | "capture" | "inject"
        self.n_out:  int  = 0          # output image token count (n_tok per frame)
        self.has_kv: bool = False

    @property
    def has_obj_kv(self) -> bool:
        return bool(self.obj_kv)

    def promote(self):
        """After each editing step: commit new_kv/new_obj_kv → kv/obj_kv."""
        if self.new_kv:
            self.kv     = dict(self.new_kv)
            self.new_kv = {}
            self.has_kv = True
        if self.new_obj_kv:
            self.obj_kv     = dict(self.new_obj_kv)
            self.new_obj_kv = {}
        self.mode = "normal"


class QwenSKAProcessor:
    """
    Drop-in replacement for QwenDoubleStreamAttnProcessor2_0 on SKA_BLOCKS.

    "normal" : fully delegates to the stored original processor — zero deviation.
    "capture": project K/V → capture scene-conditioning slice → delegate to original
               for the attention computation itself.
    "inject" : project K/V → capture scene slice → extend K/V with stored scene K/V
               → custom SDPA.  Falls back to original if RoPE or SDPA raise.
    """

    def __init__(self, block_idx: int, store: SKAStore, original_processor):
        self.block_idx = block_idx
        self.store     = store
        self.original  = original_processor   # used for normal mode and fallback

    def __call__(
        self,
        attn,
        hidden_states:              torch.Tensor,
        encoder_hidden_states:      torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        image_rotary_emb:           tuple        | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = self.store.mode

        # ── normal mode: delegate entirely — no numerical deviation ───────────
        if mode == "normal":
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )

        # ── capture / inject mode ─────────────────────────────────────────────
        B, L_img, inner_dim = hidden_states.shape
        L_txt    = encoder_hidden_states.shape[1]
        n_heads  = attn.heads
        head_dim = inner_dim // n_heads
        n_out    = self.store.n_out   # tokens per output frame

        # Project → (B, L, H, d) → norm
        img_q = attn.to_q(hidden_states).reshape(B, L_img, n_heads, head_dim)
        img_k = attn.to_k(hidden_states).reshape(B, L_img, n_heads, head_dim)
        img_v = attn.to_v(hidden_states).reshape(B, L_img, n_heads, head_dim)

        if getattr(attn, "norm_q", None) is not None: img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None: img_k = attn.norm_k(img_k)

        txt_q = attn.add_q_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)
        txt_k = attn.add_k_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)
        txt_v = attn.add_v_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)

        if getattr(attn, "norm_added_q", None) is not None: txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None: txt_k = attn.norm_added_k(txt_k)

        # ── CAPTURE (before try-block so it always succeeds even if RoPE fails)
        # Frame layout: [output(0:n_out) | scene_cond(n_out:2*n_out) | obj(2*n_out:)]
        # img_k/v are (B, L, H, d) here; permute to (B, H, n_out, d) for storage.
        # Store on CPU so captured tensors don't hold GPU memory across editing steps
        # (critical when cpu_offload is enabled; harmless otherwise).
        if n_out > 0 and L_img >= 2 * n_out:
            # Background: scene_cond frame (input reference scene)
            self.store.new_kv[self.block_idx] = (
                img_k[:, n_out:2 * n_out, :, :].permute(0, 2, 1, 3).detach().cpu(),
                img_v[:, n_out:2 * n_out, :, :].permute(0, 2, 1, 3).detach().cpu(),
            )
        if n_out > 0 and L_img >= 3 * n_out:
            # Object: obj frame (current object being inserted)
            self.store.new_obj_kv[self.block_idx] = (
                img_k[:, 2 * n_out:3 * n_out, :, :].permute(0, 2, 1, 3).detach().cpu(),
                img_v[:, 2 * n_out:3 * n_out, :, :].permute(0, 2, 1, 3).detach().cpu(),
            )

        # In capture-only mode the attention computation is handled by the original
        # processor (no injection needed, and we avoid any custom SDPA risk).
        if mode == "capture":
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )

        # ── INJECT mode: extend K/V, then custom SDPA ────────────────────────
        try:
            # RoPE on (B, L, H, d) — BEFORE head transpose
            if image_rotary_emb is not None and _apply_rope is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_q = _apply_rope(img_q, img_freqs, use_real=False)
                img_k = _apply_rope(img_k, img_freqs, use_real=False)
                txt_q = _apply_rope(txt_q, txt_freqs, use_real=False)
                txt_k = _apply_rope(txt_k, txt_freqs, use_real=False)

            # Transpose to (B, H, L, d) for SDPA
            img_q = img_q.transpose(1, 2)
            img_k = img_k.transpose(1, 2)
            img_v = img_v.transpose(1, 2)
            txt_q = txt_q.transpose(1, 2)
            txt_k = txt_k.transpose(1, 2)
            txt_v = txt_v.transpose(1, 2)

            # Extend K/V with stored background K/V (scene_cond from previous step)
            n_stored = 0
            if self.store.has_kv and self.block_idx in self.store.kv:
                s_bg_k, s_bg_v = self.store.kv[self.block_idx]   # (B, H, n_out, d)
                s_bg_k = s_bg_k.to(img_k.device, img_k.dtype)
                s_bg_v = s_bg_v.to(img_v.device, img_v.dtype)
                img_k = torch.cat([img_k, s_bg_k], dim=2)
                img_v = torch.cat([img_v, s_bg_v], dim=2)
                n_stored += s_bg_k.shape[2]

            # Extend K/V with stored object K/V (obj frame from previous step)
            if self.store.has_obj_kv and self.block_idx in self.store.obj_kv:
                s_obj_k, s_obj_v = self.store.obj_kv[self.block_idx]
                s_obj_k = s_obj_k.to(img_k.device, img_k.dtype)
                s_obj_v = s_obj_v.to(img_v.device, img_v.dtype)
                img_k = torch.cat([img_k, s_obj_k], dim=2)
                img_v = torch.cat([img_v, s_obj_v], dim=2)
                n_stored += s_obj_k.shape[2]

            # Joint attention
            joint_q = torch.cat([txt_q, img_q], dim=2)
            joint_k = torch.cat([txt_k, img_k], dim=2)
            joint_v = torch.cat([txt_v, img_v], dim=2)

            attn_mask = None
            if encoder_hidden_states_mask is not None:
                img_ones = torch.ones((B, L_img), dtype=torch.bool, device=hidden_states.device)
                kv_bool  = torch.cat([encoder_hidden_states_mask, img_ones], dim=1)
                if n_stored > 0:
                    kv_bool = torch.cat([
                        kv_bool,
                        torch.ones((B, n_stored), dtype=torch.bool, device=hidden_states.device),
                    ], dim=1)
                attn_mask = (1.0 - kv_bool.float()[:, None, None, :]) * torch.finfo(joint_q.dtype).min

            joint_out = F.scaled_dot_product_attention(
                joint_q, joint_k, joint_v,
                attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
            )   # (B, H, L_txt+L_img, d)

            txt_out = joint_out[:, :, :L_txt, :].transpose(1, 2).reshape(B, L_txt, inner_dim)
            img_out = joint_out[:, :, L_txt:L_txt + L_img, :].transpose(1, 2).reshape(B, L_img, inner_dim)

            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            txt_out = attn.to_add_out(txt_out)

            return img_out, txt_out

        except Exception:
            # RoPE/SDPA failed — fall back to original for this call.
            # Capture already populated new_kv above, so next step can inject.
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )


class _SKACallback:
    """
    callback_on_step_end: fires after step i completes, sets mode for step i+1.
    Switches from "normal" to the active phase (capture/inject) once σ < gate.
    """
    def __init__(self, store: SKAStore, inject_start: int, phase: str):
        self.store        = store
        self.inject_start = inject_start
        self.phase        = phase   # "capture" or "inject"

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if i + 1 >= self.inject_start:
            self.store.mode = self.phase
        return callback_kwargs


def _install_ska(transformer, blocks: list, store: SKAStore) -> dict:
    originals = {}
    for idx in blocks:
        blk = transformer.transformer_blocks[idx]
        orig = blk.attn.processor
        originals[idx] = orig
        blk.attn.set_processor(QwenSKAProcessor(idx, store, orig))
    return originals


def _restore_ska(transformer, originals: dict) -> None:
    for idx, proc in originals.items():
        transformer.transformer_blocks[idx].attn.set_processor(proc)


def run_qwen_ska(
    pipe,
    image,
    prompt:          str,
    seed:            int,
    num_steps:       int,
    guidance:        float,
    height:          int,
    width:           int,
    negative_prompt: str,
    ska_store:       SKAStore | None,
    ska_phase:       str      | None,   # "capture" | "inject" | None (skip)
    ska_gate:        float    = SKA_SIGMA_GATE,
) -> Image.Image:
    """
    run_qwen with optional SKA.

    ska_phase="capture" — capture scene K/V for future injection; no injection.
    ska_phase="inject"  — inject previous scene K/V + capture current scene K/V.
    ska_phase=None      — plain run_qwen (removal steps, or SKA disabled).

    After the call, ska_store.kv is updated via promote().
    """
    if ska_phase is None or ska_store is None:
        return run_qwen(pipe, image, prompt, seed, num_steps, guidance,
                        height, width, negative_prompt)

    h_tok = height // (pipe.vae_scale_factor * 2)
    w_tok = width  // (pipe.vae_scale_factor * 2)
    ska_store.n_out  = h_tok * w_tok
    ska_store.mode   = "normal"
    ska_store.new_kv = {}

    # Approximate step index where σ first drops below gate (linear FlowMatch schedule)
    inject_start = max(1, int(num_steps * (1.0 - ska_gate)) - 1)
    callback     = _SKACallback(ska_store, inject_start, ska_phase)

    originals = _install_ska(pipe.transformer, SKA_BLOCKS, ska_store)
    try:
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=image, num_inference_steps=num_steps,
            true_cfg_scale=guidance,
            height=height, width=width,
            generator=gen,
            callback_on_step_end=callback,
        ).images[0]
    finally:
        _restore_ska(pipe.transformer, originals)
        ska_store.promote()   # new_kv → kv for next step

    return result


# ── Main chain ────────────────────────────────────────────────────────────────

def run_chain(
    pipe,
    base:         Image.Image,
    edits:        List[dict],
    sketch_dir:   str,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    obj_guidance: float,
    height:       int,
    width:        int,
    out_dir:      str,
    use_ska:      bool = True,
    ska_gate:     float = SKA_SIGMA_GATE,
) -> List[Image.Image]:
    results   = [base]
    scene     = base
    obj_cache: dict = {}
    ska_store = SKAStore() if use_ska else None

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")
        tag    = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        if action == "insert":
            if name not in obj_cache:
                sketch_path = os.path.join(sketch_dir, f"{name}.png")
                if not os.path.isfile(sketch_path):
                    sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
                if not os.path.isfile(sketch_path):
                    raise FileNotFoundError(
                        f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                        f"in {sketch_dir!r}"
                    )
                print(f"  [A] Generating '{desc}' from sketch ...")
                obj_img = generate_object(
                    pipe=pipe, sketch_path=sketch_path, description=desc,
                    seed=seed, num_steps=num_steps, guidance=obj_guidance,
                    height=height, width=width,
                )
                obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))
                obj_cache[name] = obj_img
                print(f"      Saved: obj_gen_{name}.png")
            else:
                print(f"  [A] Reusing cached obj_gen_{name}.png")
                obj_img = obj_cache[name]

            qwen_image = [scene.convert("RGB"), obj_img.convert("RGB")]
            prompt = insert_prompt(desc)
            neg_p  = "blurry, distorted, low quality, watermark, text, artifacts"

            # SKA phase: first insertion → capture only; subsequent → inject + capture
            ska_phase: str | None = None
            if use_ska and ska_store is not None:
                ska_phase = "inject" if ska_store.has_kv else "capture"
            print(f"  [K] Multi-image pass: [scene, obj_ref] → {width}×{height}"
                  + (f"  SKA={ska_phase}" if ska_phase else ""))

        else:  # remove
            qwen_image = scene.convert("RGB")
            prompt = remove_prompt(desc)
            neg_p  = f"blurry, distorted, low quality, watermark, text, artifacts, {desc}"

            # No injection on removal (gives full freedom for background inpainting),
            # but still capture so the post-removal scene becomes the new reference.
            ska_phase = "capture" if (use_ska and ska_store is not None) else None
            print(f"  [K] Removal pass → {width}×{height}"
                  + (f"  SKA={ska_phase}" if ska_phase else ""))

        with open(os.path.join(out_dir, f"prompt_{tag}.txt"), "w") as f:
            f.write(prompt)
        print(f"      {prompt[:120]}...")

        next_scene = run_qwen_ska(
            pipe=pipe, image=qwen_image, prompt=prompt,
            seed=seed, num_steps=num_steps, guidance=guidance,
            height=height, width=width, negative_prompt=neg_p,
            ska_store=ska_store, ska_phase=ska_phase,
            ska_gate=ska_gate,
        )
        next_scene.save(os.path.join(out_dir, f"result_{tag}.png"))
        print(f"      Saved: result_{tag}.png")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Grid helper ────────────────────────────────────────────────────────────────

def save_grid(images, titles, path, ncols=None, figsize_per=(4, 4)):
    n     = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="No-mask reference pipeline for Qwen-Image-Edit")
    p.add_argument("--sketch_dir",      required=True)
    p.add_argument("--hf_token",        required=True)
    p.add_argument("--cache_dir",       default="./models")
    p.add_argument("--out_dir",         default="results/phase1_ref_qwen")
    p.add_argument("--model",           default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--guidance",        type=float, default=None,
                   help="CFG scale — default 1.0 for --lightning, 4.0 otherwise")
    p.add_argument("--obj_guidance",    type=float, default=None,
                   help="Object render CFG — default same as --guidance")
    p.add_argument("--num_steps",       type=int,   default=None,
                   help="Denoising steps — default 8 for --lightning, 50 otherwise")
    p.add_argument("--height",          type=int,   default=1024)
    p.add_argument("--width",           type=int,   default=1024)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--cpu_offload",     action="store_true",
                   help="Force CPU offload (auto-enabled when GPU < 65 GB)")
    p.add_argument("--base_scene",      default=None,
                   help="Path to a pre-existing base scene PNG — skips Step 0 generation")
    p.add_argument("--no_ska",          action="store_true",
                   help="Disable Scene K/V Anchoring")
    p.add_argument("--ska_gate",        type=float, default=None,
                   help="SKA sigma gate — default 0.3; consider 0.5 for 8-step Lightning")
    # Lightning distillation
    p.add_argument("--lightning",       action="store_true",
                   help="Use Lightning distilled LoRA (lightx2v/Qwen-Image-Lightning)")
    p.add_argument("--lightning_steps", type=int, default=8, choices=[4, 8],
                   help="Lightning LoRA variant: 4 or 8 steps (default: 8)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve None sentinels — Lightning changes the sensible defaults
    num_steps    = args.num_steps    if args.num_steps    is not None else (8    if args.lightning else 50)
    guidance     = args.guidance     if args.guidance     is not None else (1.0  if args.lightning else 4.0)
    obj_guidance = args.obj_guidance if args.obj_guidance is not None else guidance
    ska_gate     = args.ska_gate     if args.ska_gate     is not None else SKA_SIGMA_GATE

    print(f"\n{_SEP}")
    print(f"  phase1_ref_qwen  —  No-mask multi-image reference pipeline")
    print(f"{_SEP}")
    print(f"  Model      : {args.model}")
    if args.lightning:
        print(f"  Lightning  : ON  ({args.lightning_steps}-step LoRA from {LIGHTNING_REPO})")
    print(f"  Sketch dir : {args.sketch_dir}")
    print(f"  Output     : {args.width}×{args.height}")
    print(f"  Guidance   : scene={guidance}  obj={obj_guidance}")
    print(f"  Steps      : {num_steps}")
    print(f"  SKA        : {'OFF (--no_ska)' if args.no_ska else f'ON  blocks={SKA_BLOCKS}  σ<{ska_gate}'}")
    print(f"  Out dir    : {args.out_dir}")
    print(f"{_SEP}\n")

    print(f"Loading {args.model} ...")
    pipe = load_pipeline(
        model=args.model, hf_token=args.hf_token,
        cache_dir=args.cache_dir, device=args.device,
        cpu_offload=args.cpu_offload,
        lightning=args.lightning,
        lightning_steps=args.lightning_steps,
    )

    print("\n=== Step 0: Base scene ===")
    base_path = os.path.join(args.out_dir, "base_scene.png")
    if args.base_scene:
        base = Image.open(args.base_scene).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS)
        base.save(base_path)
        print(f"  Loaded: {args.base_scene}")
    else:
        grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
        base = run_qwen(
            pipe=pipe, image=grey, prompt=BASE_PROMPT,
            seed=args.seed, num_steps=num_steps, guidance=guidance,
            height=args.height, width=args.width,
        )
        base.save(base_path)
        print(f"  Saved: base_scene.png")

    results = run_chain(
        pipe=pipe, base=base, edits=EDITS,
        sketch_dir=args.sketch_dir,
        seed=args.seed, num_steps=num_steps,
        guidance=guidance, obj_guidance=obj_guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir,
        use_ska=not args.no_ska,
        ska_gate=ska_gate,
    )

    labels = ["base"] + [
        f"{'−' if e['action']=='remove' else '+'}{e['name']}" for e in EDITS
    ]
    save_grid(results, labels,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(results))

    print(f"\n{_SEP}")
    print(f"  Done. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
