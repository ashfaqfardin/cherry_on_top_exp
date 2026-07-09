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

from NewWork.UltimateFlux.sampler import (
    load_pipeline, generate_dual_branch, generate_masked_delta_flow,
    TIER_A, TIER_B, N_DOUBLE, N_SINGLE,
)
from NewWork.UltimateFlux.policies import (
    NonRigidPolicy,
    ObjectAdditionPolicy,
    ObjectReplacementPolicy,
    BackgroundReplacePolicy,
    FineGrainedAttrPolicy,
    ColorCtrlPolicy,
    KontextColorPolicy,
    StylePersonalizationPolicy,
)


# ───────────────────────────── Policy factory ─────────────────────────────────

def build_policy(cfg: dict):
    """Construct the right policy from a run config dict."""
    task = cfg.get("task", "non_rigid")

    if task == "non_rigid":
        return NonRigidPolicy(
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
            inject_all_single=cfg.get("inject_all_single", True),
            bg_steps_frac=tuple(cfg["bg_steps_frac"]) if cfg.get("bg_steps_frac") else (0.5, 1.0),
        )

    if task == "object_add":
        mask = None
        if cfg.get("placement_mask"):
            mask = Image.open(cfg["placement_mask"]).convert("L")
        return ObjectAdditionPolicy(
            added_word=cfg.get("added_word", None),
            placement_mask=mask,
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
            derive_step=cfg.get("derive_step", 7),
            top_k_frac=cfg.get("top_k_frac", 0.15),
        )

    if task == "object_replace":
        mask = None
        if cfg.get("mask_image"):
            mask = Image.open(cfg["mask_image"]).convert("L")
        return ObjectReplacementPolicy(
            mask=mask,
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 0.9),
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
        inject_layers_raw = cfg.get("inject_layers", None)
        if inject_layers_raw in ("color", "colour"):
            # Kontext-style colour editing: Q+K injection in all 19 double-stream
            # blocks locks spatial structure (face, car shape); 38 single-stream
            # blocks run fully free so the edit text ("blonde", "blue") drives
            # colour via V in the joint attention.  Uses generate_dual_branch.
            # Optional PFB (svd_alpha > 0): blends source principal features into
            # edit's residual for extra identity anchoring.
            return KontextColorPolicy(
                qk_layers=list(range(N_DOUBLE)),
                k_only_layers=list(range(N_DOUBLE, N_DOUBLE + N_SINGLE)),
                qk_steps_frac=tuple(cfg.get("qk_steps_frac", [0.0, 1.0])),
                k_only_steps_frac=tuple(cfg.get("k_only_steps_frac", [0.0, 1.0])),
                ss_q_steps_frac=tuple(cfg.get("ss_q_steps_frac", [0.0, 0.5])),
                svd_alpha=cfg.get("svd_alpha", 0.0),
                svd_layers=cfg.get("svd_layers", [1]),
                svd_steps_frac=tuple(cfg.get("svd_steps_frac", [0.0, 0.25])),
            )
        elif inject_layers_raw == "double_stream":
            # Lock all 19 double-stream (joint text-image) blocks; 38 single-stream free.
            # key_only=True (default): inject only K into double-stream blocks.
            #   K locks SPATIAL POSITIONS (structure preserved, no forehead drift).
            #   V stays target-conditioned ("blonde"/"blue") → full colour change.
            # key_only=False: inject K+V (stronger structure lock, weaker colour change).
            return FineGrainedAttrPolicy(
                inject_layers=list(range(N_DOUBLE)),  # _DOUBLE_STREAM: 0-18
                inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
                key_only=cfg.get("key_only", True),
            )
        elif inject_layers_raw == "tier_a":
            inject_layers = TIER_A          # K+V — breed/shape change
        else:
            inject_layers = None            # default → _PRESERVE_LAYERS, K+V
        return FineGrainedAttrPolicy(
            inject_layers=inject_layers,
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
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
    name               = cfg.get("name", "output")
    source_prompt      = cfg.get("source_prompt", cfg.get("prompt", ""))
    edit_prompt        = cfg.get("edit_prompt",   cfg.get("prompt", ""))
    seed               = cfg.get("seed", 42)
    num_steps          = cfg.get("num_steps", 28)
    guidance           = cfg.get("guidance_scale", 3.5)
    height             = cfg.get("height", 1024)
    width              = cfg.get("width",  1024)
    max_seq_len        = cfg.get("max_sequence_length", 512)
    save_intermediates = cfg.get("save_intermediates", False)
    intermediate_every = cfg.get("intermediate_every", 4)
    delta_scale        = cfg.get("delta_scale", 2.0)
    delta_start_step   = cfg.get("delta_start_step", None)

    policy = build_policy(cfg)

    # Always create run_dir when saving anything (finals or intermediates).
    run_dir = os.path.join(out_dir, name)
    if save_images or save_intermediates:
        os.makedirs(run_dir, exist_ok=True)

    print(f"\n[UltimateFlux] '{name}' | task={cfg.get('task','non_rigid')} | seed={seed}")
    print(f"  source_prompt: {source_prompt}")
    print(f"  edit_prompt:   {edit_prompt}")

    # ColorCtrlPolicy (legacy) uses masked delta-flow.
    # All other policies (including KontextColorPolicy) use dual-branch attention injection.
    if isinstance(policy, ColorCtrlPolicy) and not isinstance(policy, KontextColorPolicy):
        print(f"  [route] generate_masked_delta_flow (delta_scale={delta_scale})")
        src_img, edit_img = generate_masked_delta_flow(
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
            delta_scale=delta_scale,
            delta_start_step=delta_start_step,
            save_intermediates=save_intermediates,
            intermediate_out_dir=run_dir if save_intermediates else None,
            intermediate_every=intermediate_every,
        )
    else:
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
            save_intermediates=save_intermediates,
            intermediate_out_dir=run_dir if save_intermediates else None,
            intermediate_every=intermediate_every,
        )

    if save_images:
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
    p.add_argument("--save_intermediates", action="store_true", default=False,
                   help="Save source/edit/compare intermediate step grids alongside final images")
    p.add_argument("--intermediate_every", type=int, default=4,
                   help="Capture a frame every N denoising steps (--save_intermediates)")

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
    p.add_argument("--inject_steps_frac", type=float, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Non-rigid / attr_edit: fraction of denoising steps to apply injection. "
                        "Default depends on task (non_rigid: 0.0 1.0, attr_edit: 0.0 1.0).")
    p.add_argument("--inject_all_single", action="store_true", default=True,
                   help="Non-rigid: also inject K,V at ALL single-stream layers to lock background. Default True.")
    p.add_argument("--no_inject_all_single", dest="inject_all_single", action="store_false",
                   help="Non-rigid: only inject at TIER_A layers (disable background locking).")
    p.add_argument("--bg_steps_frac", type=float, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Non-rigid: step window for background-lock (all-single-stream) injection. "
                        "Default 0.5 1.0 — starts at halfway so early steps can establish the new pose. "
                        "Lower START for stronger background lock; raise for more pose freedom.")
    p.add_argument("--inject_layers", default=None,
                   choices=["color", "double_stream", "tier_a"],
                   help="attr_edit mode: color=Kontext Q+K injection (colour change), double_stream=K+V in 19 joint blocks, tier_a=breed/shape change")
    p.add_argument("--key_only", action="store_true", default=False,
                   help="double_stream mode: inject K only (not V) — locks spatial layout without locking colour. Default True for double_stream via config.")
    p.add_argument("--reweight_scale", type=float, default=1.0,
                   help="color mode: multiply image→colour-word attention scores by this factor pre-softmax (§3.5). 1.0=disabled; try 3.0–5.0 for strong colour change.")
    p.add_argument("--ds_key_inject", action="store_true", default=False,
                   help="color mode: also inject K in double-stream blocks to prevent face-structure drift when qk_frac=0.")
    p.add_argument("--top_k_frac",  type=float, default=0.2,
                   help="ColorCtrl: fraction of image tokens treated as editing region (default 0.2)")
    p.add_argument("--qk_frac",    type=float, default=1.0,
                   help="ColorCtrl: fraction of steps for structure preservation / v-v score injection")
    p.add_argument("--v_frac",     type=float, default=1.0,
                   help="ColorCtrl: fraction of steps for colour preservation / V masking")
    p.add_argument("--color_word", default=None,
                   help="ColorCtrl: colour word in edit prompt (e.g. 'blonde', 'blue'); "
                        "focuses the editing-region mask on that word's T5 tokens")
    p.add_argument("--chunk_size", type=int, default=4,
                   help="ColorCtrl: heads per chunk in manual attention (reduce to 2/1 for OOM)")
    p.add_argument("--color_sam2", action="store_true", default=False,
                   help="color mode: use SAM2 point-prompted segmentation for the editing mask "
                        "instead of attention topk. Requires: pip install sam2")
    p.add_argument("--color_sam2_model", default="facebook/sam2-hiera-large",
                   help="HuggingFace SAM2 model ID for color mask segmentation")
    p.add_argument("--mask_build_step", type=int, default=5,
                   help="ColorCtrl: denoising step at which to build the editing-region mask "
                        "(steps 0..N-1 run freely; mask built at N, ColorCtrl from N..end). "
                        "Larger = more accurate mask but less structure locked early.")
    p.add_argument("--qk_steps_frac", type=float, nargs=2, default=[0.0, 1.0],
                   help="color mode: [start end] fraction of steps for double-stream Q+K injection.")
    p.add_argument("--k_only_steps_frac", type=float, nargs=2, default=[0.0, 1.0],
                   help="color mode: step fraction for single-stream K injection (always-on position lock).")
    p.add_argument("--ss_q_steps_frac", type=float, nargs=2, default=[0.0, 0.5],
                   help="color mode: step fraction for single-stream Q injection (identity lock). "
                        "Default first 50%% of steps. Raise end toward 1.0 for tighter identity; "
                        "lower end toward 0.0 for stronger colour.")
    p.add_argument("--svd_alpha",   type=float, default=0.0,
                   help="color mode: PFB alpha for optional SVD block hook (0 = disabled). "
                        "Enable (e.g. 1.0) to blend source structural features into edit's residual.")
    p.add_argument("--svd_layers",  type=int, nargs="+", default=[1],
                   help="color mode: block indices for PFB SVD hook (default: [1])")
    p.add_argument("--svd_steps_frac", type=float, nargs=2, default=[0.0, 0.25],
                   help="color mode: step fraction for PFB SVD hook (default first 25%)")
    p.add_argument("--color_structure_frac", type=float, default=0.0,
                   help="Unused; kept for backward compat")
    p.add_argument("--inject_frac", type=float, default=0.5,
                   help="Unused; kept for backward compat")
    p.add_argument("--pfb_step",   type=int,   default=3,
                   help="Unused; kept for backward compat")
    p.add_argument("--pfb_alpha",   type=float, default=1.0,
                   help="Unused; kept for backward compat")
    p.add_argument("--delta_scale", type=float, default=2.0,
                   help="Unused; kept for backward compat (legacy masked-delta-flow)")
    p.add_argument("--delta_start_step", type=int, default=None,
                   help="Unused; kept for backward compat (legacy masked-delta-flow)")
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
        "inject_steps_frac":    args.inject_steps_frac,
        "inject_all_single":    args.inject_all_single,
        "bg_steps_frac":        args.bg_steps_frac,
        "inject_layers":        args.inject_layers,
        "key_only":             args.key_only,
        "reweight_scale":       args.reweight_scale,
        "ds_key_inject":        args.ds_key_inject,
        "top_k_frac":           args.top_k_frac,
        "qk_frac":              args.qk_frac,
        "v_frac":               args.v_frac,
        "color_word":           args.color_word,
        "chunk_size":           args.chunk_size,
        "mask_build_step":      args.mask_build_step,
        "color_sam2":           args.color_sam2,
        "color_sam2_model":     args.color_sam2_model,
        "qk_steps_frac":        args.qk_steps_frac,
        "k_only_steps_frac":    args.k_only_steps_frac,
        "ss_q_steps_frac":      args.ss_q_steps_frac,
        "svd_alpha":            args.svd_alpha,
        "svd_layers":           args.svd_layers,
        "svd_steps_frac":       args.svd_steps_frac,
        "color_structure_frac": args.color_structure_frac,
        "inject_frac":          args.inject_frac,
        "pfb_step":             args.pfb_step,
        "pfb_alpha":            args.pfb_alpha,
        "delta_scale":          args.delta_scale,
        "delta_start_step":     args.delta_start_step,
        "seed":               args.seed,
        "num_steps":          args.num_steps,
        "guidance_scale":     args.guidance_scale,
        "height":             args.height,
        "width":              args.width,
        "save_intermediates": args.save_intermediates,
        "intermediate_every": args.intermediate_every,
    }
    run_single(pipe, single_cfg, args.out_dir, args.save_images, args.device)


if __name__ == "__main__":
    main()
