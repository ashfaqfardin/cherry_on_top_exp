"""
UltimateFlux — unified training-free editing on FLUX.1-dev.

Synthesises StableFlow, FluxSpace, FreeFlux, and SVD-Style into a single
dual-branch pipeline with paper-validated layer sets for FLUX.1-dev
(§6–7 of Pipeline_Plan.md).

Supported tasks
---------------
  non_rigid          Change pose/action; preserve appearance (Task 1)
  object_replace     Swap an object in a masked region        (Task 3)
  bg_replace         Regenerate background; keep foreground   (Task 4)
  attr_edit          Disentangled single-attribute edit       (Task 5)
  style              Reference-image style personalization    (Task 7)

Usage — single run
------------------
  python NewWork/UltimateFlux/run_ultimateflux.py \\
      --task non_rigid \\
      --source_prompt "a bird perched on a branch" \\
      --edit_prompt   "a bird flying from the branch" \\
      --seed 42 --device cuda --save_images

  python NewWork/UltimateFlux/run_ultimateflux.py \\
      --task style \\
      --source_prompt "A cat" \\
      --edit_prompt   "A cat" \\
      --style_image   inputs/watercolor_ref.png \\
      --seed 42 --device cuda --save_images

Usage — config file (batch)
----------------------------
  python NewWork/UltimateFlux/run_ultimateflux.py \\
      --config prompts/ultimateflux_demo.json \\
      --hf_token "$HF_TOKEN" --device cuda --save_images
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image

# Allow running from repo root or from this file's directory.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from NewWork.UltimateFlux.sampler import load_pipeline, generate_dual_branch, TIER_A, TIER_B
from NewWork.UltimateFlux.policies import (
    NonRigidPolicy,
    ObjectAdditionPolicy,
    ObjectReplacementPolicy,
    BackgroundReplacePolicy,
    FineGrainedAttrPolicy,
    StylePersonalizationPolicy,
)


# ───────────────────────────── Policy factory ─────────────────────────────────

def build_policy(cfg: dict):
    """Construct the right policy from a run config dict."""
    task = cfg.get("task", "non_rigid")

    if task == "non_rigid":
        return NonRigidPolicy(
            inject_steps_frac=tuple(cfg.get("inject_steps_frac", [0.0, 0.8])),
        )

    if task == "object_add":
        mask = None
        if cfg.get("placement_mask"):
            mask = Image.open(cfg["placement_mask"]).convert("L")
        return ObjectAdditionPolicy(
            added_word=cfg.get("added_word", None),
            placement_mask=mask,
            inject_steps_frac=tuple(cfg.get("inject_steps_frac", [0.0, 1.0])),
            derive_step=cfg.get("derive_step", 7),
            top_k_frac=cfg.get("top_k_frac", 0.15),
        )

    if task == "object_replace":
        mask = None
        if cfg.get("mask_image"):
            mask = Image.open(cfg["mask_image"]).convert("L")
        return ObjectReplacementPolicy(
            mask=mask,
            inject_steps_frac=tuple(cfg.get("inject_steps_frac", [0.0, 0.9])),
        )

    if task == "bg_replace":
        fg_mask = None
        if cfg.get("fg_mask_image"):
            fg_mask = Image.open(cfg["fg_mask_image"]).convert("L")
        return BackgroundReplacePolicy(
            fg_mask=fg_mask,
            use_sam2=cfg.get("use_sam2", True),
            sam2_model_id=cfg.get("sam2_model_id", "facebook/sam2-hiera-large"),
        )

    if task == "attr_edit":
        _HOTSPOT = [1, 2, 4, 26, 30, 54, 55]
        inject_layers = cfg.get("inject_layers", None)
        if inject_layers == "hotspot":
            inject_layers = _HOTSPOT        # colour/texture change: structure only
        elif inject_layers == "tier_a":
            inject_layers = TIER_A          # breed/shape change: appearance lock
        elif inject_layers is not None:
            inject_layers = None            # unrecognised → default _PRESERVE_LAYERS
        return FineGrainedAttrPolicy(
            inject_layers=inject_layers,
            inject_steps_frac=tuple(cfg.get("inject_steps_frac", [0.0, 1.0])),
        )

    if task == "style":
        style_img = None
        if cfg.get("style_image"):
            style_img = Image.open(cfg["style_image"]).convert("RGB")
        return StylePersonalizationPolicy(
            style_image=style_img,
            alpha=cfg.get("pfb_alpha", 1.0),
            sac_steps_frac=tuple(cfg.get("sac_steps_frac", [0.0, 1.0])),
            pfb_steps_frac=tuple(cfg.get("pfb_steps_frac", [0.0, 1.0])),
        )

    raise ValueError(
        f"Unknown task '{task}'. "
        "Choose from: non_rigid, object_replace, bg_replace, attr_edit, style"
    )


# ───────────────────────────── Single run ─────────────────────────────────────

def run_single(pipe, cfg: dict, out_dir: str, save_images: bool, device: str):
    name          = cfg.get("name", "output")
    source_prompt = cfg.get("source_prompt", cfg.get("prompt", ""))
    edit_prompt   = cfg.get("edit_prompt",   cfg.get("prompt", ""))
    seed          = cfg.get("seed", 42)
    num_steps     = cfg.get("num_steps", 28)
    guidance      = cfg.get("guidance_scale", 3.5)
    height        = cfg.get("height", 1024)
    width         = cfg.get("width",  1024)
    max_seq_len   = cfg.get("max_sequence_length", 512)

    policy = build_policy(cfg)

    print(f"\n[UltimateFlux] '{name}' | task={cfg.get('task','non_rigid')} | seed={seed}")
    print(f"  source_prompt: {source_prompt}")
    print(f"  edit_prompt:   {edit_prompt}")

    src_img, edit_img = generate_dual_branch(
        pipe=pipe,
        policy=policy,
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        max_sequence_length=max_seq_len,
        device=device,
    )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        src_path  = os.path.join(run_dir, "source.png")
        edit_path = os.path.join(run_dir, "edit.png")
        src_img.save(src_path)
        edit_img.save(edit_path)
        print(f"  Saved → {src_path}")
        print(f"  Saved → {edit_path}")

        # Side-by-side comparison.
        # For style task: [style_ref | source | styled] so the reference is visible.
        panels = [src_img, edit_img]
        style_ref_path = cfg.get("style_image")
        if cfg.get("task") == "style" and style_ref_path:
            ref = Image.open(style_ref_path).convert("RGB").resize(
                (src_img.width, src_img.height), Image.LANCZOS)
            panels = [ref, src_img, edit_img]

        comp_w = sum(p.width for p in panels)
        comp_h = max(p.height for p in panels)
        comp   = Image.new("RGB", (comp_w, comp_h))
        x = 0
        for panel in panels:
            comp.paste(panel, (x, 0))
            x += panel.width
        comp_path = os.path.join(run_dir, "compare.png")
        comp.save(comp_path)
        print(f"  Saved → {comp_path}")

    return src_img, edit_img


# ───────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UltimateFlux unified editing pipeline")

    # Model / infra
    p.add_argument("--model_path",  default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/ultimateflux")
    p.add_argument("--save_images", action="store_true")

    # Config file (batch mode)
    p.add_argument("--config", default=None, help="Path to JSON config for batch runs")

    # Single-run overrides
    p.add_argument("--task",          default="non_rigid",
                   choices=["non_rigid", "object_add", "object_replace", "bg_replace", "attr_edit", "style"])
    p.add_argument("--name",          default="output")
    p.add_argument("--source_prompt", default="")
    p.add_argument("--edit_prompt",   default="")
    p.add_argument("--prompt",        default=None, help="Sets both source and edit prompt")
    p.add_argument("--style_image",   default=None)
    p.add_argument("--added_word",    default=None, help="Word for new object (object_add), e.g. 'vase'")
    p.add_argument("--placement_mask",default=None, help="Placement region mask (object_add)")
    p.add_argument("--mask_image",    default=None, help="Object region mask (object_replace)")
    p.add_argument("--fg_mask_image", default=None, help="Foreground mask (bg_replace)")
    p.add_argument("--use_sam2",      action="store_true", default=True,
                   help="Auto-segment foreground with SAM2 when no fg_mask_image given (bg_replace)")
    p.add_argument("--no_sam2",       dest="use_sam2", action="store_false",
                   help="Disable SAM2; fall back to TIER_A global injection (bg_replace)")
    p.add_argument("--sam2_model_id", default="facebook/sam2-hiera-large",
                   help="HuggingFace SAM2 model ID (bg_replace)")
    p.add_argument("--inject_layers", default=None,
                   choices=["hotspot", "tier_a"],
                   help="Layer set for attr_edit: hotspot=colour/texture change, tier_a=shape/breed change")
    p.add_argument("--pfb_alpha",     type=float, default=1.0)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--num_steps",     type=int, default=28)
    p.add_argument("--guidance_scale",type=float, default=3.5)
    p.add_argument("--height",        type=int, default=1024)
    p.add_argument("--width",         type=int, default=1024)

    return p.parse_args()


def main():
    args = parse_args()

    # ── Load pipeline ──
    print(f"[UltimateFlux] Loading {args.model_path} …")
    pipe = load_pipeline(
        model_path=args.model_path,
        hf_token=args.hf_token,
        device=args.device,
        cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print(f"[UltimateFlux] Model loaded on {args.device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Config file (batch) ──
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

        global_cfg = config.get("global", {})
        runs       = config.get("runs", [])

        for run_cfg in runs:
            merged = {**global_cfg, **run_cfg}
            run_single(pipe, merged, args.out_dir, args.save_images, args.device)

        return

    # ── Single run ──
    source_prompt = args.prompt if args.prompt else args.source_prompt
    edit_prompt   = args.prompt if args.prompt else args.edit_prompt

    single_cfg = {
        "name":          args.name,
        "task":          args.task,
        "source_prompt": source_prompt,
        "edit_prompt":   edit_prompt,
        "style_image":    args.style_image,
        "added_word":     args.added_word,
        "placement_mask": args.placement_mask,
        "mask_image":     args.mask_image,
        "fg_mask_image":  args.fg_mask_image,
        "use_sam2":       args.use_sam2,
        "sam2_model_id":  args.sam2_model_id,
        "inject_layers":  args.inject_layers,
        "pfb_alpha":     args.pfb_alpha,
        "seed":          args.seed,
        "num_steps":     args.num_steps,
        "guidance_scale":args.guidance_scale,
        "height":        args.height,
        "width":         args.width,
    }
    run_single(pipe, single_cfg, args.out_dir, args.save_images, args.device)


if __name__ == "__main__":
    main()
