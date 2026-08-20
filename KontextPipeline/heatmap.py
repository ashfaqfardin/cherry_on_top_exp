from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image

from .kv_inject import _TIER_A

_HEATMAP_STOP = {"a", "an", "the", "with", "on", "in", "of", "and", "or",
                 "no", "not", "very", "some", "this", "that", "is", "are",
                 "for", "to", "at", "its", "into", "from"}


def _find_desc_token_indices(pipe, prompt: str, description: str) -> List[int]:
    """
    Return T5 token indices for every content word in description.
    Averaging heatmaps across multiple tokens gives a stronger and more
    specific placement signal than a single word.
    """
    tokenizer   = pipe.tokenizer_2
    ids         = tokenizer.encode(prompt, add_special_tokens=False)
    content     = [w.lower().strip(".,!?") for w in description.split()
                   if w.lower().strip(".,!?") not in _HEATMAP_STOP and len(w) >= 3]

    found: List[int] = []
    for i, tid in enumerate(ids):
        tok = tokenizer.decode([tid]).strip().lower()
        for word in content:
            if (tok == word
                    or (len(tok) >= 3 and tok in word)
                    or (len(word) >= 3 and word in tok)):
                if i not in found:
                    found.append(i)
                break

    if not found:
        print(f"    [HEATMAP] No content tokens for '{description}' — using idx 0")
        return [0]
    print(f"    [HEATMAP] Content-word tokens for '{description}': indices {found[:10]}")
    return found


class _HeatmapCapture:
    """
    DAAM-style attention heatmap capture.

    Computes raw Q_gen · K_text[obj_indices] / √d at each TIER_A double-stream
    layer and accumulates the per-gen-token score across all content-word tokens.
    """
    def __init__(self, n_gen: int, obj_token_indices: List[int]):
        self.n_gen             = n_gen
        self.obj_token_indices = obj_token_indices
        self._layer = 0; self._step = 0
        self.heatmap: Optional[np.ndarray] = None
        self._count  = 0

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
        if is_double and self._layer in _TIER_A:
            g_lo = txt_len; g_hi = txt_len + self.n_gen
            if q.shape[2] >= g_hi:
                scale = hd ** -0.5
                q_gen = q[:,:,g_lo:g_hi,:]
                acc   = None; n_hit = 0
                for obj_idx in self.obj_token_indices:
                    if obj_idx < txt_len:
                        score = (q_gen @ k[:,:,obj_idx:obj_idx+1,:].transpose(-1,-2)
                                 ).squeeze(-1) * scale
                        s_np  = score.float().mean(dim=(0,1)).detach().cpu().numpy()
                        acc   = s_np if acc is None else acc + s_np
                        n_hit += 1
                if acc is not None and n_hit > 0:
                    if self.heatmap is None:
                        self.heatmap = np.zeros(self.n_gen, dtype=np.float32)
                    self.heatmap += acc / n_hit
                    self._count  += 1
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out     = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


def _vlm_predict_placement(
    vlm_model, vlm_processor,
    scene: Image.Image, description: str,
) -> Optional[str]:
    """
    Ask the VLM where the object naturally belongs in this scene.
    Returns a short placement phrase like 'on the kitchen counter' or None.
    """
    device = next(vlm_model.parameters()).device

    w, h = scene.size
    if max(w, h) > 768:
        s = 768 / max(w, h)
        scene_q = scene.resize((int(w * s), int(h * s)), Image.LANCZOS).convert("RGB")
    else:
        scene_q = scene.convert("RGB")

    query = (
        f"Where in this scene would a {description} naturally be placed? "
        f"Reply with ONLY a short phrase (2-6 words) starting with 'on', 'against', or 'near'. "
        f"Examples: 'on the floor', 'on the wooden table', 'against the brick wall', "
        f"'on the stone path', 'on the kitchen counter'. "
        f"Give ONLY the phrase — no explanation, no punctuation."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": scene_q},
            {"type": "text",  "text": query},
        ],
    }]

    try:
        text   = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[text], images=[scene_q], padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out_ids = vlm_model.generate(**inputs, max_new_tokens=24, do_sample=False)
        answer = vlm_processor.decode(
            out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower().rstrip(".,!?")

        print(f"    [VLM PLACE] '{answer}'")

        for prefix in ("on ", "against ", "near ", "beside ", "under ", "above "):
            if answer.startswith(prefix):
                return " ".join(answer.split()[:6])

        for marker in ("on the", "on a", "against the", "near the", "beside the",
                       "under the", "above the"):
            idx = answer.find(marker)
            if idx >= 0:
                fragment = " ".join(answer[idx:].split()[:6]).rstrip(".,!?")
                print(f"    [VLM PLACE] Extracted: '{fragment}'")
                return fragment

        print(f"    [VLM PLACE] Unparseable ('{answer}') — no anchor used")
        return None
    except Exception as e:
        print(f"    [VLM PLACE] Error: {e} — no anchor used")
        return None


@torch.no_grad()
def run_heatmap_pass(
    pipe, scene: Image.Image, prompt: str, description: str,
    seed: int, height: int, width: int,
    n_steps: int = 6,
    occupied_mask: Optional[np.ndarray] = None,
    anchor: Optional[str] = None,
) -> np.ndarray:
    """
    Short n_steps Kontext forward pass → DAAM attention heatmap (h_lat, w_lat) float32 [0,1].

    anchor: optional VLM placement phrase appended to prompt so the T5 encoding
            biases the object tokens spatially toward the predicted surface.
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    h_lat  = height // (vae_sf * 2)
    w_lat  = width  // (vae_sf * 2)
    n_gen  = h_lat * w_lat

    if anchor:
        hm_prompt = prompt.rstrip(".! ") + f", {anchor}."
        print(f"    [HEATMAP] Anchor: '{anchor}'")
    else:
        hm_prompt = prompt
        print(f"    [HEATMAP] No anchor — using description tokens only")

    orig_procs        = pipe.transformer.attn_processors
    obj_token_indices = _find_desc_token_indices(pipe, hm_prompt, description)

    cap = _HeatmapCapture(n_gen=n_gen, obj_token_indices=obj_token_indices)
    pipe.transformer.set_attn_processor(cap)

    def _step_cb(pr, si, ts, ck):
        cap._step = si + 1; cap._layer = 0
        return ck

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=scene, prompt=hm_prompt,
         num_inference_steps=n_steps, guidance_scale=2.5,
         height=height, width=width, max_sequence_length=512,
         generator=gen, output_type="latent",
         callback_on_step_end=_step_cb,
         callback_on_step_end_tensor_inputs=[])
    pipe.transformer.set_attn_processor(orig_procs)

    if cap.heatmap is None or cap._count == 0:
        print("    [HEATMAP] No maps captured — returning uniform heatmap")
        return np.ones((h_lat, w_lat), dtype=np.float32)

    hm = (cap.heatmap / cap._count).reshape(h_lat, w_lat)

    if occupied_mask is not None:
        occ = cv2.resize((occupied_mask > 127).astype(np.uint8),
                         (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
        occ = cv2.dilate(occ, np.ones((5, 5), np.uint8), iterations=2)
        hm  = hm * (1.0 - occ.astype(np.float32))

    lo, hi = hm.min(), hm.max()
    if hi - lo > 1e-6:
        hm = (hm - lo) / (hi - lo)
    peak = np.unravel_index(hm.argmax(), hm.shape)
    print(f"    [HEATMAP] {cap._count} maps, {len(obj_token_indices)} tokens  "
          f"peak=({peak[1]},{peak[0]})")
    return hm.astype(np.float32)


def heatmap_to_mask(
    heatmap:    np.ndarray,
    height:     int,
    width:      int,
    percentile: float = 72,
    blur_ksize: int   = 11,
    blur_sigma: float = 4.0,
    min_frac:   float = 0.04,
    max_frac:   float = 0.45,
) -> np.ndarray:
    """
    (h_lat, w_lat) float heatmap → uint8 pixel mask (H, W).

    Used for BLD callback and bbox tracking.
    K/V injection uses the raw heatmap directly as soft weights (see run.py).
    """
    h_lat, w_lat = heatmap.shape
    blurred = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), blur_sigma)
    binary  = (blurred >= np.percentile(blurred, percentile)).astype(np.uint8)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary  = (labels == largest).astype(np.uint8)

    frac = binary.mean()
    if frac < min_frac:
        cy, cx = np.unravel_index(blurred.argmax(), blurred.shape)
        r = max(3, int(min(h_lat, w_lat) * 0.15))
        binary = np.zeros_like(binary)
        binary[max(0,cy-r):min(h_lat,cy+r), max(0,cx-r):min(w_lat,cx+r)] = 1
        print(f"    [HEATMAP] Mask too small ({frac*100:.1f}%) — peak neighbourhood")
    elif frac > max_frac:
        p2     = percentile + (100 - percentile) * 0.5
        binary = (blurred >= np.percentile(blurred, p2)).astype(np.uint8)
        print(f"    [HEATMAP] Mask too large ({frac*100:.1f}%) — re-threshold p={p2:.0f}")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.vstack(contours)
        if len(all_pts) >= 4:
            rect    = cv2.minAreaRect(all_pts)
            box     = cv2.boxPoints(rect).astype(np.int32)
            filled  = np.zeros_like(binary)
            cv2.fillPoly(filled, [box], 1)
            binary  = filled
            cx, cy  = rect[0]
            print(f"    [HEATMAP] Rotated rect: centre=({cx:.0f},{cy:.0f})  "
                  f"size=({rect[1][0]:.0f}×{rect[1][1]:.0f})  angle={rect[2]:.1f}°")

    mask_np = np.array(
        Image.fromarray(binary * 255).resize((width, height), Image.NEAREST)
    ).astype(np.uint8)
    print(f"    [HEATMAP] Final mask: {int((mask_np>127).mean()*100)}% of pixels")
    return mask_np
