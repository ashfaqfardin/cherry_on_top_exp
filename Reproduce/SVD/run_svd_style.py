"""
CLI runner for SVD-based style personalization on Infinity-2B.

Paper: "A Training-Free Style-Personalization via SVD-Based Feature Decomposition"
       DGIST 2025

Run from the repo root (e:/Cherry_on_top/):

    python Reproduce/SVD/run_svd_style.py \\
        --infinity_path  /path/to/infinity_2b.pth \\
        --vae_path       /path/to/infinity_vae_d32.pth \\
        --t5_path        google/flan-t5-xl \\
        --style_image    inputs/watercolor_ref.jpg \\
        --prompt         "a cat sitting on a windowsill in watercolor style" \\
        --seed 0 --height 1024 --width 1024 \\
        --device cuda --cache_dir ./models --save_images

Or pass a config file:

    python Reproduce/SVD/run_svd_style.py \\
        --config prompts/reproduce_svd_style.json \\
        --device cuda --cache_dir ./models --save_images
"""

import argparse
import json
import os
import sys

# ── make repo root importable ────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# also ensure the Infinity submodule is importable (users clone it alongside
# this repo or install it; we look for it under <root>/Infinity if present)
_INF = os.path.join(_ROOT, "Infinity")
if os.path.isdir(_INF) and _INF not in sys.path:
    sys.path.insert(0, _INF)

import torch
from PIL import Image

from Reproduce.SVD.svd_style_pipeline import generate_styled, load_model


# ──────────────────────────── helpers ────────────────────────────────


def _save(img: Image.Image, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    img.save(path)
    print(f"  saved → {path}")


def _run_one(
    infinity, vae, text_tokenizer, text_encoder, scale_schedule,
    *,
    name: str,
    style_image_path: str,
    prompt: str,
    pfb_alpha: float,
    cfg: float,
    tau: float,
    top_k: int,
    top_p: float,
    cfg_insertion_layer: int,
    seed: int,
    height: int,
    width: int,
    device: str,
    out_dir: str,
    save_images: bool,
) -> Image.Image:
    print(f"\n[{name}]")
    print(f"  style image : {style_image_path}")
    print(f"  prompt      : {prompt}")

    style_img = Image.open(style_image_path).convert("RGB")

    img = generate_styled(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        style_image=style_img,
        prompt=prompt,
        scale_schedule=scale_schedule,
        pfb_alpha=pfb_alpha,
        cfg=cfg,
        tau=tau,
        top_k=top_k,
        top_p=top_p,
        cfg_insertion_layer=cfg_insertion_layer,
        seed=seed,
        height=height,
        width=width,
        device=device,
    )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        _save(img, run_dir, "generated")

    return img


# ──────────────────────────── CLI ────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVD style personalization (Infinity-2B)")

    # ── model ────────────────────────────────────────────────────────
    p.add_argument("--infinity_path", type=str, default="",
                   help="Path to Infinity-2B checkpoint (.pth).  "
                        "Falls back to cache_dir/infinity_2b.pth")
    p.add_argument("--vae_path", type=str, default="",
                   help="Path to BSQ-VAE d32 checkpoint (.pth).  "
                        "Falls back to cache_dir/infinity_vae_d32.pth")
    p.add_argument("--t5_path", type=str, default="google/flan-t5-xl",
                   help="HuggingFace model ID or local path for T5 text encoder")
    p.add_argument("--cache_dir", type=str, default="./models",
                   help="Directory for cached model weights")

    # ── single-run arguments (ignored when --config is given) ─────────
    p.add_argument("--style_image", type=str, default="",
                   help="Path to reference style image")
    p.add_argument("--prompt", type=str, default="",
                   help='Text prompt, e.g. "a cat in watercolor style"')
    p.add_argument("--name", type=str, default="output",
                   help="Output sub-folder name")

    # ── method hyper-parameters ───────────────────────────────────────
    p.add_argument("--pfb_alpha", type=float, default=1.0,
                   help="SVD exponential reweighting factor α (paper default 1.0)")
    p.add_argument("--cfg", type=float, default=3.0,
                   help="Classifier-free guidance scale")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=900,
                   help="Top-k for sampling")
    p.add_argument("--top_p", type=float, default=0.97,
                   help="Top-p (nucleus) sampling threshold")
    p.add_argument("--cfg_insertion_layer", type=int, default=-5,
                   help="Layer index at which CFG is inserted (negative = from end)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # ── runtime ──────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="results/svd_style",
                   help="Root directory for output images")
    p.add_argument("--save_images", action="store_true",
                   help="Write generated images to disk")
    p.add_argument("--config", type=str, default="",
                   help="Path to JSON config file (overrides single-run args)")

    return p.parse_args()


def _resolve_path(path: str, fallback_dir: str, fallback_name: str) -> str:
    if path:
        return path
    return os.path.join(fallback_dir, fallback_name)


def main() -> None:
    args = parse_args()

    inf_ckpt = _resolve_path(args.infinity_path, args.cache_dir, "infinity_2b.pth")
    vae_ckpt = _resolve_path(args.vae_path,      args.cache_dir, "infinity_vae_d32.pth")

    print("[Loading models …]")
    infinity, vae, text_tokenizer, text_encoder, scale_schedule = load_model(
        model_path=inf_ckpt,
        vae_path=vae_ckpt,
        t5_path=args.t5_path,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # ── build run list ───────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg_json = json.load(f)
        global_cfg = cfg_json.get("global", {})
        runs = cfg_json.get("runs", [])
    else:
        if not args.style_image or not args.prompt:
            raise ValueError("Provide --style_image and --prompt, or --config")
        global_cfg = {}
        runs = [
            {
                "name":         args.name,
                "style_image":  args.style_image,
                "prompt":       args.prompt,
            }
        ]

    def _g(key, default):
        """Resolve: run-level → global → CLI arg → default."""
        return global_cfg.get(key, getattr(args, key, default))

    for run in runs:
        _run_one(
            infinity, vae, text_tokenizer, text_encoder, scale_schedule,
            name=run.get("name", "output"),
            style_image_path=run["style_image"],
            prompt=run["prompt"],
            pfb_alpha=run.get("pfb_alpha",          _g("pfb_alpha",          1.0)),
            cfg=run.get("cfg",                       _g("cfg",                3.0)),
            tau=run.get("tau",                       _g("tau",                1.0)),
            top_k=run.get("top_k",                   _g("top_k",              900)),
            top_p=run.get("top_p",                   _g("top_p",              0.97)),
            cfg_insertion_layer=run.get(
                "cfg_insertion_layer",               _g("cfg_insertion_layer", -5)),
            seed=run.get("seed",                     _g("seed",               0)),
            height=run.get("height",                 _g("height",             1024)),
            width=run.get("width",                   _g("width",              1024)),
            device=args.device,
            out_dir=args.out_dir,
            save_images=args.save_images,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
