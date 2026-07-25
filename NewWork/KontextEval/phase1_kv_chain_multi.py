"""
Phase 1 — K/V Chain Multi: Generalized N-Step Multi-Object Preservation

The problem with the standard Kontext chain in multi-step editing
-----------------------------------------------------------------
At step K, Kontext freely regenerates the entire scene from img_{K-1}.
Objects added in steps 1..K-1 exist only as pixels in that reference image.
Kontext MAY shift, recolor, or erase prior objects when the "add object K"
prompt draws its attention elsewhere.

The compound invariant that fixes it
-------------------------------------
At step K, the reference IS img_{K-1}, which already contains every object
added so far. If we inject ref K/V into every token EXCEPT where object K
needs to appear, all prior content (every object + the base scene) is
automatically anchored to img_{K-1}.

  background_mask = ~target_mask_K

  reference = img_{K-1}  →  carries full edit history
  injection at background_mask  →  protects everything that existed before K

This means:
  step 2 reference = img_1  → protects object 1
  step 3 reference = img_2  → protects objects 1 AND 2
  step K reference = img_{K-1}  → protects objects 1..K-1

No per-object mask management is needed. The background = ~target rule is
sufficient at every step.

Two-pass design per step
------------------------
We can't know the target_mask BEFORE running inference (we don't know exactly
where Kontext will place the new object). So each step runs twice:

  PROBE PASS (args.probe_steps, no injection):
    Run standard Kontext to find the natural object placement.
    fast — only needs to show WHERE the object lands.

  INJECT PASS (args.num_steps, with injection):
    target_mask = pixel_to_token_mask(img_{K-1}, img_probe, threshold)
    background_mask = ~target_mask
    Re-run with K/V injection at background_mask.
    The probe's placement is an APPROXIMATION; the inject pass may shift
    the object slightly. That is fine — the mask just needs to free the
    right general area.

Step 1 also benefits from this pattern: the reference is the base scene,
so injecting base K/V into non-bicycle tokens keeps the room stable while
only the bicycle region generates freely from the prompt.

Edit list (default: bicycle → vase)
--------------------------------------
Configurable via EDITS at the top of the file or --config JSON.
Add or remove objects freely — the pipeline handles N steps identically.

Metrics
-------
  stability[K]: mean abs pixel diff in object K's region between
    - step K output  (when object K was first added)
    - FINAL output   (img_N)
  Lower = object K is better preserved across all subsequent edits.

  Reported for both baseline (no injection) and kv_multi chain.

Comparison
----------
  baseline:  standard Kontext chain, no injection
  kv_multi:  probe-then-inject at every step with background=~target

Usage
-----
python NewWork/KontextEval/phase1_kv_chain_multi.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_kv_chain_multi

Add a third object:
    edit EDITS list below, or pass --config my_edits.json

  my_edits.json format:
  [
    {"name": "bicycle", "prompt": "Add a yellow bicycle leaning against the wall."},
    {"name": "vase",    "prompt": "Add a white ceramic vase on the coffee table."},
    {"name": "lamp",    "prompt": "Add a floor lamp next to the sofa."}
  ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate

_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)
from kontext_injection import (
    TIER_A, TIER_ALL,
    ZoneMasks, InjectionState,
    install_processor, set_determinism,
)

# ============================================================
# Default edit list — change or extend for your scene
# ============================================================

EDITS: List[dict] = [
    {
        "name": "bicycle",
        "prompt": (
            "Add a yellow bicycle leaning against the wall on the left side. "
            "Keep the rest of the room exactly the same."
        ),
    },
    {
        "name": "vase",
        "prompt": (
            "Add a white ceramic vase with flowers on the coffee table. "
            "Keep the rest of the room exactly the same."
        ),
    },
]

BASE_PROMPT = "A modern living room with a sofa and a wooden coffee table."

TIER_A_LAYERS = list(TIER_A)
ALL_LAYERS     = list(TIER_ALL)


# ============================================================
# Mask utilities
# ============================================================

def pixel_to_token_mask(img_a: Image.Image, img_b: Image.Image,
                        h_lat: int, w_lat: int,
                        threshold: float = 40.0) -> np.ndarray:
    """
    Flat bool (n_gen=h_lat*w_lat,) — True where |img_b - img_a| > threshold.
    Pixel diff is computed in image space then downsampled to the latent
    token grid (one token = 16×16 image pixels for FLUX Kontext).
    """
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    diff_down = diff_img.resize((w_lat, h_lat), Image.BILINEAR)
    return (np.array(diff_down).astype(np.float32) >= threshold).reshape(-1)


def overlay(img: Image.Image, flat_mask: np.ndarray,
            h_lat: int, w_lat: int,
            color=(255, 120, 0), alpha=0.4) -> Image.Image:
    """Orange tint where mask==True (bilinear up from token grid)."""
    H, W = img.size[1], img.size[0]
    token_2d = flat_mask.reshape(h_lat, w_lat).astype(np.uint8) * 255
    pixel_mask = np.array(
        Image.fromarray(token_2d, "L").resize((W, H), Image.NEAREST)
    ) > 127
    arr = np.array(img).astype(float)
    out = arr.copy()
    out[pixel_mask] = arr[pixel_mask] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def region_diff(img_a: Image.Image, img_b: Image.Image,
                flat_mask: np.ndarray, h_lat: int, w_lat: int) -> float:
    """Mean absolute pixel diff (0–255) in the masked region."""
    H, W = img_a.size[1], img_a.size[0]
    token_2d = flat_mask.reshape(h_lat, w_lat).astype(np.uint8) * 255
    px_mask = np.array(
        Image.fromarray(token_2d, "L").resize((W, H), Image.NEAREST)
    ) > 127
    diff = np.abs(np.array(img_a).astype(float) - np.array(img_b).astype(float)).mean(axis=2)
    return float(diff[px_mask].mean()) if px_mask.any() else 0.0


# ============================================================
# Generation helpers
# ============================================================

def run_standard(pipe, canvas: Image.Image, prompt: str,
                 seed: int, num_steps: int, guidance: float,
                 height: int, width: int) -> Image.Image:
    return generate(
        pipe, prompt, canvas,
        seed=seed, num_steps=num_steps,
        guidance_scale=guidance,
        height=height, width=width,
    )


@torch.no_grad()
def run_injected(pipe, canvas: Image.Image, prompt: str,
                 background_mask: np.ndarray,
                 seed: int, num_steps: int, guidance: float,
                 height: int, width: int,
                 strength: float, cutoff: float,
                 vital_layers: List[int],
                 max_seq_len: int = 512,
                 device: str = "cuda") -> Image.Image:
    """
    Kontext denoising with K/V injection at background_mask tokens.

    background_mask (flat bool, n_gen): tokens where the scene already
      exists and must be preserved.  ~background_mask = target region where
      the new object can generate freely.

    At TIER_A layers during the first `cutoff` fraction of steps:
      K_gen[background] ← (1-s)*K_gen + s*K_ref   (from same-call ref slice)
      V_gen[background] ← (1-s)*V_gen + s*V_ref
    """
    h_lat  = height // 16
    w_lat  = width  // 16
    n_gen  = h_lat * w_lat

    target_mask = np.logical_not(background_mask)

    state = InjectionState(
        mode="edit",
        vital_layers=set(vital_layers),
        n_gen=n_gen,
        n_ref=n_gen,
        cutoff_frac=(0.0, cutoff),
        strength=strength,
        n_steps=num_steps,
    )
    state.zones = ZoneMasks(
        background=background_mask.astype(bool),
        shell=np.zeros(n_gen, dtype=bool),
        target=target_mask.astype(bool),
    ).to_device(device)

    install_processor(pipe, state, max_sequence_length=max_seq_len)
    generator = set_determinism(seed)
    result = pipe(
        image=canvas,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=height, width=width,
        max_sequence_length=max_seq_len,
        generator=generator,
        output_type="pil",
    )
    return result.images[0]


# ============================================================
# N-step chains
# ============================================================

def run_baseline_chain(pipe, base: Image.Image, edits: List[dict],
                       seed: int, num_steps: int, guidance: float,
                       height: int, width: int) -> List[Image.Image]:
    """Standard Kontext chain — no injection at any step."""
    imgs = [base]
    for edit in edits:
        img = run_standard(pipe, imgs[-1], edit["prompt"],
                           seed, num_steps, guidance, height, width)
        imgs.append(img)
    return imgs


def run_kv_multi_chain(pipe, base: Image.Image, edits: List[dict],
                       h_lat: int, w_lat: int,
                       seed: int, num_steps: int, probe_steps: int,
                       guidance: float, height: int, width: int,
                       strength: float, cutoff: float,
                       vital_layers: List[int],
                       threshold: float, out_dir: str,
                       device: str = "cuda") -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    N-step chain with background=~target K/V injection at every step.

    Returns
    -------
    imgs      : [base, img_1, img_2, ...] length = len(edits)+1
    obj_masks : [target_mask_1, target_mask_2, ...] length = len(edits)
                Each mask marks the region where object K was placed.
    """
    imgs       = [base]
    obj_masks  = []

    for i, edit in enumerate(edits):
        name     = edit["name"]
        prompt   = edit["prompt"]
        img_prev = imgs[-1]

        print(f"\n  [step {i+1}] {name}")

        # --- Probe: find natural placement of the new object ---
        print(f"    probe ({probe_steps} steps) …")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"kv_step{i+1}_{name}_probe.png"))

        # Target mask = tokens where the probe differs from the previous image
        target_mask = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        obj_masks.append(target_mask)

        pct_target = target_mask.mean() * 100
        print(f"    target region: {pct_target:.1f}% of tokens  "
              f"(threshold={threshold})")

        # Save target overlay on the previous image so user can verify
        overlay(img_prev, target_mask, h_lat, w_lat,
                color=(0, 160, 255), alpha=0.45).save(
            os.path.join(out_dir, f"kv_step{i+1}_{name}_target_overlay.png")
        )
        # Background overlay
        bg_mask = np.logical_not(target_mask)
        overlay(img_prev, bg_mask, h_lat, w_lat,
                color=(255, 140, 0), alpha=0.25).save(
            os.path.join(out_dir, f"kv_step{i+1}_{name}_background_overlay.png")
        )

        # --- Inject pass: protect everything except the target region ---
        print(f"    inject ({num_steps} steps, s={strength}, cutoff={cutoff}) …")
        background_mask = np.logical_not(target_mask)
        img_curr = run_injected(
            pipe, img_prev, prompt, background_mask,
            seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
            strength=strength, cutoff=cutoff,
            vital_layers=vital_layers, device=device,
        )
        img_curr.save(os.path.join(out_dir, f"kv_step{i+1}_{name}_result.png"))

        imgs.append(img_curr)

    return imgs, obj_masks


# ============================================================
# Grid / comparison helpers
# ============================================================

def save_grid(images, titles, path, ncols=None):
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=8)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",     required=True)
    p.add_argument("--cache_dir",    default="./models")
    p.add_argument("--out_dir",      default="results/phase1_kv_chain_multi")
    p.add_argument("--config",       default=None,
                   help="JSON file with list of {name, prompt} dicts. "
                        "Overrides the built-in EDITS list.")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_steps",    type=int,   default=28)
    p.add_argument("--probe_steps",  type=int,   default=10,
                   help="Denoising steps for the probe pass (speed vs accuracy). "
                        "10 is usually enough to reveal object placement.")
    p.add_argument("--guidance",     type=float, default=2.5)
    p.add_argument("--strength",     type=float, default=0.7,
                   help="K/V injection weight s. "
                        "s=0: no injection (= baseline). s=1: full lock.")
    p.add_argument("--cutoff",       type=float, default=0.6,
                   help="Inject during first CUTOFF fraction of denoising steps. "
                        "0.6 = first 60%% of steps (structure-forming phase).")
    p.add_argument("--threshold",    type=float, default=40.0,
                   help="Pixel diff threshold (0-255) for target token mask.")
    p.add_argument("--all_layers",   action="store_true",
                   help="Use all 57 layers instead of TIER_A (13 layers). "
                        "Expected to produce pixel-identical copy (no edit).")
    p.add_argument("--height",       type=int,   default=1024)
    p.add_argument("--width",        type=int,   default=1024)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)
        print(f"Loaded {len(edits)} edits from {args.config}")

    h_lat = args.height // 16
    w_lat = args.width  // 16
    vital_layers = ALL_LAYERS if args.all_layers else TIER_A_LAYERS

    print(f"Edit sequence: {[e['name'] for e in edits]}")
    print(f"Injection: s={args.strength}, cutoff={args.cutoff}, "
          f"layers={'ALL_57' if args.all_layers else 'TIER_A'}")
    print(f"Probe steps: {args.probe_steps}  |  Inject steps: {args.num_steps}")

    print("\nLoading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ----------------------------------------------------------
    # Base scene (step 0)
    # ----------------------------------------------------------
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(pipe, grey, BASE_PROMPT,
                        args.seed, args.num_steps, args.guidance,
                        args.height, args.width)
    base.save(os.path.join(args.out_dir, "step0_base.png"))

    # ----------------------------------------------------------
    # Baseline chain (no injection)
    # ----------------------------------------------------------
    print("\n=== BASELINE chain (no injection) ===")
    baseline_imgs = run_baseline_chain(
        pipe, base, edits,
        args.seed, args.num_steps, args.guidance, args.height, args.width,
    )
    for i, edit in enumerate(edits):
        baseline_imgs[i + 1].save(
            os.path.join(args.out_dir, f"baseline_step{i+1}_{edit['name']}.png")
        )

    # ----------------------------------------------------------
    # K/V multi chain (probe + inject at every step)
    # ----------------------------------------------------------
    print("\n=== K/V MULTI chain (background=~target injection) ===")
    kv_imgs, obj_masks = run_kv_multi_chain(
        pipe, base, edits,
        h_lat=h_lat, w_lat=w_lat,
        seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
        guidance=args.guidance, height=args.height, width=args.width,
        strength=args.strength, cutoff=args.cutoff,
        vital_layers=vital_layers,
        threshold=args.threshold,
        out_dir=args.out_dir, device=args.device,
    )

    # ----------------------------------------------------------
    # Stability metrics
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("STABILITY TABLE  (how much each object changed in later steps)")
    print(f"{'='*65}")
    print(f"  {'object':<12}  {'baseline Δ':>12}  {'kv_multi Δ':>12}  {'improvement':>12}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    stability_lines = [
        "object_stability: mean abs pixel diff in object region vs FINAL image\n"
        "lower = object better preserved across all subsequent edits\n\n"
        f"{'object':<12}  {'baseline_diff':>14}  {'kv_multi_diff':>14}  {'improvement%':>13}\n"
        f"{'-'*12}  {'-'*14}  {'-'*14}  {'-'*13}\n"
    ]

    for i, (edit, obj_mask) in enumerate(zip(edits, obj_masks)):
        name = edit["name"]
        # Compare the image at step i+1 (when object was added) vs final output
        # baseline
        b_step  = baseline_imgs[i + 1]
        b_final = baseline_imgs[-1]
        b_diff  = region_diff(b_step, b_final, obj_mask, h_lat, w_lat)
        # kv_multi
        k_step  = kv_imgs[i + 1]
        k_final = kv_imgs[-1]
        k_diff  = region_diff(k_step, k_final, obj_mask, h_lat, w_lat)

        pct = (b_diff - k_diff) / max(b_diff, 1e-6) * 100
        marker = " ← IMPROVED" if k_diff < b_diff else " ← WORSE"
        print(f"  {name:<12}  {b_diff:>12.2f}  {k_diff:>12.2f}  {pct:>+11.1f}%{marker}")
        stability_lines.append(
            f"{name:<12}  {b_diff:>14.2f}  {k_diff:>14.2f}  {pct:>+12.1f}%\n"
        )

    print(f"\n  Δ = mean abs pixel diff (0-255) in the object's token region")
    print(f"  LOWER = object more stable across later edits")
    print()

    with open(os.path.join(args.out_dir, "stability.txt"), "w") as f:
        f.writelines(stability_lines)
    print(f"  Saved: stability.txt")

    # ----------------------------------------------------------
    # KEY_RESULT: step-by-step comparison grid
    # ----------------------------------------------------------
    n_steps = len(edits)

    # Row 1: baseline steps 0..N
    # Row 2: kv_multi steps 0..N (results only, not probes)
    all_imgs   = [base] + baseline_imgs[1:] + [base] + kv_imgs[1:]
    all_titles = (
        ["Base"] + [f"Baseline step {i+1}\n{edits[i]['name']}" for i in range(n_steps)]
        + ["Base"] + [f"KV-Multi step {i+1}\n{edits[i]['name']}" for i in range(n_steps)]
    )
    save_grid(
        all_imgs, all_titles,
        os.path.join(args.out_dir, "KEY_RESULT_comparison.png"),
        ncols=n_steps + 1,
    )
    print(f"  Saved: KEY_RESULT_comparison.png  ({2} rows × {n_steps+1} cols)")

    # Final side-by-side (baseline final vs kv_multi final)
    save_grid(
        [baseline_imgs[-1], kv_imgs[-1]],
        [f"Baseline final  (no injection)",
         f"KV-Multi final  (s={args.strength}, cutoff={args.cutoff})"],
        os.path.join(args.out_dir, "KEY_RESULT_final.png"),
    )
    print(f"  Saved: KEY_RESULT_final.png")

    # ----------------------------------------------------------
    # What to check
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("WHAT TO CHECK")
    print(f"{'='*65}")
    print(f"""
KEY_RESULT_comparison.png  ← most important
  Top row: baseline chain — watch for objects drifting in later steps.
  Bottom row: kv_multi chain — prior objects should be more stable.
  Look at EACH object in EACH subsequent step:
    Does the bicycle change between step 1 and step 2?  (top vs bottom row)
    If bottom row bicycle is more stable → injection is working.

KEY_RESULT_final.png
  Left: final baseline image (all objects added, no protection).
  Right: final kv_multi image (all objects protected via injection).
  SUCCESS = right image has clearer/more accurate versions of all objects,
            while the new object (last step) is still correctly placed.

kv_step*_target_overlay.png  (BLUE tint = target/new-object region)
  Verify the target mask covers the new object and NOT prior objects.
  If target_mask is too large (covers prior objects too):
    → increase --threshold (try 50-60)
  If target_mask is too small (misses the new object):
    → decrease --threshold (try 25-30), or increase --probe_steps

kv_step*_background_overlay.png  (ORANGE tint = background/protected region)
  This is everything being frozen by injection.
  It should cover: base scene + all prior objects.
  If the background region is too large (covers the new object area):
    → the new object may be suppressed. Reduce --threshold or --strength.

stability.txt
  baseline_diff vs kv_multi_diff for each object.
  A positive improvement% = kv_multi preserved that object better.
  If kv_multi is WORSE: the injection is interfering with the edit.
    Try: lower --strength (0.5), shorter --cutoff (0.4), higher --threshold.

probe images (kv_step*_probe.png)
  These show where Kontext naturally places each object without injection.
  The target mask is derived from these. If the probe looks wrong
  (wrong object placement), increase --probe_steps (try 15-20).
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
