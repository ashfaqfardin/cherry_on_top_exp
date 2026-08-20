"""
run.py — incremental object insertion / removal pipeline entry point.

Flow per insert:
  Stage A    : sketch → LoRA FLUX → obj_img
  Stage DEPTH: estimate_depth(scene) → find_floor_bbox → mask_np
  Stage HEAT : DAAM heatmap pass → soft token weights
  Stage COL  : build_collage_scene → canvas
  Stage K    : run_with_collage_kv_injection (+ BLD latent callback)
  Stage BCG  : pixel-wise object mask tracked from result diff

Flow per remove:
  Stage DEPTH: reuse tracked insertion bbox or detect_object_on_floor
  Stage COL  : build_removal_collage (Telea inpaint)
  Stage K    : run_with_collage_kv_injection (+ BLD latent callback)
  Stage BCG  : pixel-space background restore (removal only)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .loader   import load_kontext_pipeline, load_vlm
from .utils    import run_standard, save_grid
from .sketch   import generate_from_sketch, LORA_ID
from .mask_ops import (_compute_obj_mask, _rect_mask_from_bbox,
                       _token_zone_from_mask_np)
from .depth    import (load_depth_model, estimate_depth,
                       find_floor_bbox, detect_object_on_floor)
from .heatmap  import run_heatmap_pass, heatmap_to_mask, _vlm_predict_placement
from .kv_inject import (run_with_collage_kv_injection, _make_bcg_latent_callback,
                        run_with_feature_delta_injection)
from .collage  import (build_collage_scene, build_removal_collage, _collage_obj_mask)


# ── Edit list ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",         "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers", "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",              "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",              "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",         "action": "remove"},
]

BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)

_SEP = "═" * 60


# ── Main chain ────────────────────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    depth_model,
    depth_processor,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool  = True,
    kv_mode:        str   = "collage",   # "collage" | "delta"
    obj_strength:   float = 0.5,
    bg_strength:    float = 0.8,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.6),
    heatmap_steps:  int   = 6,
    vlm_model                          = None,
    vlm_processor                      = None,
) -> List[Image.Image]:
    results  = [base]
    scene    = base
    _placed:           Dict[str, Tuple[int, int, int, int]] = {}
    per_object_masks:  Dict[str, np.ndarray]                = {}

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        print(f"  [DEPTH] Estimating scene depth ...")
        depth_np = estimate_depth(depth_model, depth_processor, scene)
        if depth_np is not None:
            Image.fromarray((depth_np * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"depth_{i+1}_{name}_{action}.png")
            )

        if action == "insert":
            sketch_path = os.path.join(sketch_dir, f"{name}.png")
            if not os.path.isfile(sketch_path):
                sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
            if not os.path.isfile(sketch_path):
                raise FileNotFoundError(
                    f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                    f"in {sketch_dir!r}"
                )

            print(f"  [A] Generating '{desc}' from sketch ...")
            obj_img = generate_from_sketch(
                pipe=pipe, sketch_path=sketch_path, description=desc,
                seed=seed, num_steps=num_steps, guidance=lora_guidance,
                height=height, width=width, lora_id=lora_id, device=device,
            )
            obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))

            blend_p = f"A photorealistic room with a {desc} placed naturally."

            placement_anchor: Optional[str] = None
            if vlm_model is not None:
                print(f"  [VLM] Predicting placement for '{desc}' ...")
                placement_anchor = _vlm_predict_placement(
                    vlm_model, vlm_processor, scene, desc
                )

            _occ = np.zeros((height, width), dtype=np.uint8)
            for _m in per_object_masks.values():
                _occ = np.clip(
                    _occ.astype(np.int32) + _m.astype(np.int32) * 255, 0, 255
                ).astype(np.uint8)

            print(f"  [PLACE] Attention heatmap for '{desc}' ...")
            heatmap = run_heatmap_pass(
                pipe=pipe, scene=scene, prompt=blend_p, description=desc,
                seed=seed, height=height, width=width, n_steps=heatmap_steps,
                occupied_mask=_occ, anchor=placement_anchor,
            )
            hm_vis = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            Image.fromarray(cv2.cvtColor(hm_vis, cv2.COLOR_BGR2RGB)).save(
                os.path.join(out_dir, f"heatmap_{name}.png")
            )

            # Soft per-token injection weights from DAAM heatmap (already [0,1]).
            # Binary rotated-rectangle mask is still used for BLD and bbox tracking.
            _heatmap_weights = heatmap.reshape(-1).astype(np.float32)
            mask_np = heatmap_to_mask(heatmap, height, width)
            Image.fromarray(mask_np).save(os.path.join(out_dir, f"mask_pred_{name}.png"))

            ys, xs = np.where(mask_np > 127)
            if len(ys) > 0:
                bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            else:
                bx1, by1 = 0, height // 4
                bx2, by2 = width, 3 * height // 4
            _placed[name] = (bx1, by1, bx2, by2)
            with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
                f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

            ref_mask = _compute_obj_mask(obj_img)
            if ref_mask.sum() < 50:
                print("         Fallback: centre-crop silhouette")
                ref_mask = np.zeros((height, width), dtype=np.uint8)
                ref_mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 1

            print(f"  [COL] Building collage (heatmap-guided) ...")
            collage_scene, _ = build_collage_scene(
                scene=scene, obj_img=obj_img, ref_mask=ref_mask,
                target_bbox=(bx1, by1, bx2, by2), collage_mode="full",
            )
            collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
            obj_mask_har = _collage_obj_mask(collage_scene, scene)

        else:  # remove
            if name in _placed:
                bx1, by1, bx2, by2 = _placed[name]
                print(f"  [PLACE] Reusing tracked insertion bbox: "
                      f"x=[{bx1},{bx2}] y=[{by1},{by2}]")
            else:
                print(f"  [DETECT] Depth-anomaly object detection ...")
                bx1, by1, bx2, by2 = detect_object_on_floor(depth_np, height, width)

            if name in per_object_masks:
                mask_np = (per_object_masks[name] * 255).astype(np.uint8)
                print(f"  [PLACE] Using pixel-wise mask for removal of '{name}'")
            else:
                mask_np = _rect_mask_from_bbox(bx1, by1, bx2, by2, height, width)
            Image.fromarray(mask_np).save(
                os.path.join(out_dir, f"mask_pred_remove_{name}.png")
            )
            with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
                f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

            print(f"  [COL] Building removal collage (Telea inpaint) ...")
            collage_scene = build_removal_collage(
                scene=scene, mask_np=mask_np, height=height, width=width,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_remove_{name}.png"))

            obj_mask_har = (mask_np > 127).astype(np.uint8)
            blend_p = (
                f"{BASE_PROMPT} "
                f"The {desc} has been removed. "
                f"Seamless wooden floor and white walls fill the area. "
                f"No object, clean empty room."
            )
            _placed.pop(name, None)
            per_object_masks.pop(name, None)

        # Stage K: Kontext integration
        step_tag = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"
        print(f"  [K] Kontext integration pass  kv={use_kv} mode={kv_mode}  action={action} ...")
        with open(os.path.join(out_dir, f"prompt_{step_tag}.txt"), "w") as f:
            f.write(blend_p)

        pipe_dtype = next(pipe.transformer.parameters()).dtype
        bcg_cb = _make_bcg_latent_callback(
            scene=scene, mask_np=mask_np, pipe=pipe,
            height=height, width=width, device=device, dtype=pipe_dtype,
        )

        if use_kv:
            target_zone = _token_zone_from_mask_np(mask_np, height, width, pipe)
            _tw = _heatmap_weights if action == "insert" else None
            if kv_mode == "delta" and action == "insert":
                next_scene = run_with_feature_delta_injection(
                    pipe=pipe, scene=scene, prompt=blend_p,
                    obj_collage=collage_scene,
                    target_zone=target_zone,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                    bcg_callback=bcg_cb, target_weights=_tw,
                )
            else:
                next_scene = run_with_collage_kv_injection(
                    pipe=pipe, canvas=collage_scene, prompt=blend_p,
                    target_zone=target_zone,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                    bcg_callback=bcg_cb, target_weights=_tw,
                )
        else:
            next_scene = run_standard(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width, bcg_callback=bcg_cb,
            )

        result_path = os.path.join(out_dir, f"result_{step_tag}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        if action == "insert":
            _r   = np.array(next_scene.convert("RGB"), dtype=np.int32)
            _s   = np.array(scene.convert("RGB"),      dtype=np.int32)
            _diff = np.abs(_r - _s).max(axis=2)
            _px  = (_diff > 15).astype(np.uint8)
            _px  = cv2.morphologyEx(_px, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            _px  = cv2.morphologyEx(_px, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
            per_object_masks[name] = _px
            print(f"    [MASK] '{name}' pixel mask: {int(_px.sum())} px "
                  f"({_px.mean()*100:.1f}% of frame)")

        _COLORS = [(0, 220, 0), (0, 80, 255), (255, 60, 0),
                   (255, 200, 0), (0, 220, 220), (200, 0, 255)]
        _overlay = np.array(next_scene.convert("RGB"), dtype=np.float32)
        for _ci, (_nm, _m) in enumerate(per_object_masks.items()):
            _c = np.array(_COLORS[_ci % len(_COLORS)], dtype=np.float32)
            _overlay[_m > 0] = 0.45 * _overlay[_m > 0] + 0.55 * _c
        Image.fromarray(np.clip(_overlay, 0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"masks_overlay_{step_tag}.png")
        )

        if action == "remove":
            print(f"  [BCG] Background restore (removal) ...")
            bcg_mask  = cv2.dilate(obj_mask_har, np.ones((21, 21), np.uint8), iterations=3)
            bcg_mask  = cv2.GaussianBlur(bcg_mask.astype(np.float32), (41, 41), 15)
            result_np = np.array(next_scene.convert("RGB"), dtype=np.float32)
            scene_np  = np.array(scene.convert("RGB"),      dtype=np.float32)
            m3        = bcg_mask[:, :, None]
            bcg_np    = np.clip(result_np * m3 + scene_np * (1.0 - m3), 0, 255).astype(np.uint8)
            next_scene = Image.fromarray(bcg_np)
            bcg_path   = os.path.join(out_dir, f"result_bcg_{step_tag}.png")
            next_scene.save(bcg_path)
            print(f"      Saved: {bcg_path}")

        scene = next_scene
        results.append(scene)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AnyDoor collage + FLUX Kontext — mask-free via depth + DAAM heatmap."
    )
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/kontext_pipeline")
    p.add_argument("--depth_model",
                   default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description, action}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5)
    p.add_argument("--heatmap_steps", type=int,   default=6)
    p.add_argument("--kv_injection",  action="store_true")
    p.add_argument("--kv_mode",       default="collage", choices=["collage", "delta"],
                   help="collage: K/V injection from collage reference (default). "
                        "delta: null-referenced feature-delta injection into residual stream.")
    p.add_argument("--obj_strength",  type=float, default=0.5)
    p.add_argument("--bg_strength",   type=float, default=0.8)
    p.add_argument("--cutoff_frac",   default="0.0,0.6")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--vlm_model",     default=None)
    p.add_argument("--vlm_device",    default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)

    print(f"\n{_SEP}")
    print(f"  KontextPipeline — mask-free incremental editing")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Depth model  : {args.depth_model}")
    print(f"  VLM model    : {args.vlm_model or 'none (description tokens only)'}")
    print(f"  KV injection : {args.kv_injection}"
          + (f"  mode={args.kv_mode}  obj={args.obj_strength}  bg={args.bg_strength}"
             f"  cutoff={args.cutoff_frac}" if args.kv_injection else ""))
    print(f"  Guidance     : {args.guidance}")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading depth model ...")
    depth_model, depth_processor, _ = load_depth_model(
        model_id=args.depth_model, cache_dir=args.cache_dir,
    )

    vlm_model = vlm_processor = None
    if args.vlm_model:
        vlm_model, vlm_processor = load_vlm(
            args.vlm_model, cache_dir=args.cache_dir, device=args.vlm_device,
        )

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    _cf = [float(x) for x in args.cutoff_frac.split(",")]
    cutoff_frac: Tuple[float, float] = (_cf[0], _cf[1])

    results = run_collage_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        depth_model=depth_model, depth_processor=depth_processor,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_kv=args.kv_injection,
        kv_mode=args.kv_mode,
        obj_strength=args.obj_strength,
        bg_strength=args.bg_strength,
        cutoff_frac=cutoff_frac,
        heatmap_steps=args.heatmap_steps,
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
    )

    all_imgs = results
    all_lbls = ["base"] + [f"{e['name']} ({e.get('action','insert')})" for e in edits]
    save_grid(all_imgs, all_lbls,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
