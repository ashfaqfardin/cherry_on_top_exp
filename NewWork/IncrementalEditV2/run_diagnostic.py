"""
Norm-map diagnostic — measures whether the attention output norm localises to
the target object, and when the Otsu gate fires.

Subclasses SequentialEditor's denoising loop so the pipeline code is unchanged.
Saves per-step heatmaps and a summary grid, prints a step table to stdout.

Usage (edit 1, no prior objects):
    python NewWork/IncrementalEditV2/run_diagnostic.py \
        --hf_token hf_... \
        --prompt "a cozy living room with a red armchair" \
        --object "a red armchair" \
        --cache_dir ./models \
        --out_dir results/diagnostic_edit1

Usage (edit 2, with a prior committed image to test BCG):
    python NewWork/IncrementalEditV2/run_diagnostic.py \
        --hf_token hf_... \
        --prompt "a cozy living room with a red armchair and a sleeping cat" \
        --object "a sleeping cat" \
        --prior_image results/diagnostic_edit1/output.png \
        --cache_dir ./models \
        --out_dir results/diagnostic_edit2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make the package importable when run as a script from any working directory.
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",     required=True)
    p.add_argument("--prompt",       required=True)
    p.add_argument("--object",       required=True, dest="object_phrase")
    p.add_argument("--prior_image",  default=None,
                   help="PNG from the previous edit — used to test BCG blend")
    p.add_argument("--out_dir",      default="results/diagnostic")
    p.add_argument("--num_steps",    type=int, default=28)
    p.add_argument("--height",       type=int, default=1024)
    p.add_argument("--width",        type=int, default=1024)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--cache_dir",    default="./models")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--collect_blocks", type=int, nargs="+", default=None,
                   help="Override which dual-stream blocks feed the norm signal "
                        "(default: 8-14). Try e.g. --collect_blocks 4 5 6 7 8 9")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _save_step_png(norm_vec, ph, pw, step_idx, phase, otsu_result, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fired, mask, thr, var = otsu_result
    nmap = np.array(norm_vec).reshape(ph, pw)

    cols = 2 if fired else 1
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
    if cols == 1:
        axes = [axes]

    ax0 = axes[0]
    im = ax0.imshow(nmap, cmap="hot")
    ax0.set_title(f"step {step_idx:02d} [{phase}]  var={var:.5f}", fontsize=9)
    ax0.axis("off")
    fig.colorbar(im, ax=ax0, fraction=0.046)

    if fired:
        ax1 = axes[1]
        ax1.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax1.set_title(f"Otsu mask  thr={thr:.3f}  tokens={int(mask.sum())}/{ph*pw}",
                      fontsize=9)
        ax1.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step_idx:02d}.png"), dpi=100,
                bbox_inches="tight")
    plt.close(fig)


def _save_summary_grid(records, ph, pw, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    if n == 0:
        return
    cols = 7
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).flatten()

    for i, rec in enumerate(records):
        ax = axes[i]
        nmap = np.array(rec["norm_vec"]).reshape(ph, pw)
        ax.imshow(nmap, cmap="hot")
        fired_str = f"F{rec['mask_tokens']}" if rec["fired"] else "·"
        ax.set_title(f"s{rec['step']} {rec['phase'][:3]} {fired_str}",
                     fontsize=7)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "norm_summary.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary grid → {path}")


# ---------------------------------------------------------------------------
# Diagnostic run (mirrors SequentialEditor.add_object with extra recording)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostic_edit(editor, prompt, object_phrase, seed, out_dir):
    from flux_seqedit.masking import otsu_gate, refine_mask, soft_prior_fallback_mask
    from flux_seqedit.torch_adapters import (
        norm_vector_to_patch_map, phase_for_step, occupied_mask_to_token_tensor,
    )

    cfg = editor.cfg
    s   = editor.state
    s.reset_collection()
    s.step_index  = 0
    s.obj_token_span = editor._object_token_span(prompt, object_phrase)
    editor._bcg_noise = None

    occupied = editor.memory.occupied_union()
    device   = editor.pipe.transformer.device
    s.occupied_token_idx = (
        occupied_mask_to_token_tensor(occupied, device) if occupied.any() else None
    )

    prior_soft = editor._build_soft_prior(occupied)
    gen        = torch.Generator(device=device).manual_seed(seed)
    latents, latent_kwargs = editor._prepare_latents(prompt, gen)
    timesteps  = latent_kwargs["timesteps"]

    locked_mask = None
    records     = []

    print(f"\n{'step':>4}  {'phase':>5}  {'var':>8}  {'fired':>5}  {'tokens':>6}  {'thr':>6}")
    print("-" * 46)

    for i, t in enumerate(timesteps):
        s.step_index = i
        phase = phase_for_step(i, cfg.num_steps, cfg.early_frac, cfg.late_frac)
        s.apply_prior = (phase == "early" and s.occupied_token_idx is not None)

        noise_pred = editor._transformer_step(latents, t, latent_kwargs)
        latents    = editor._scheduler_step(noise_pred, t, latents, i)

        nrm = s.aggregate_norm()
        if nrm is not None:
            pmap = norm_vector_to_patch_map(nrm, editor.ph, editor.pw)
            fired, mask, thr, var = otsu_gate(pmap, cfg.otsu_confidence)
            tokens = int(mask.sum()) if fired else 0

            rec = {
                "step": i, "phase": phase, "fired": fired,
                "var": var, "thr": thr, "mask_tokens": tokens,
                "norm_vec": nrm.cpu().numpy().copy(),
                "mask": mask,
            }
            records.append(rec)

            fired_label = "YES" if fired else "no"
            print(f"{i:>4}  {phase:>5}  {var:>8.5f}  {fired_label:>5}  "
                  f"{tokens:>6}  {thr:>6.3f}")

            _save_step_png(nrm.cpu().numpy(), editor.ph, editor.pw,
                           i, phase, (fired, mask, thr, var), out_dir)

            if fired and locked_mask is None:
                locked_mask = refine_mask(mask)

        s.reset_collection()

        if (locked_mask is None
                and i >= cfg.otsu_hard_cutoff_frac * cfg.num_steps):
            locked_mask = soft_prior_fallback_mask(prior_soft)
            print(f"[step {i:02d}] hard-fallback triggered")

        if locked_mask is not None and editor._prev_latent is not None:
            if editor._bcg_noise is None:
                editor._bcg_noise = torch.randn_like(editor._prev_latent)
            latents = editor._bcg_blend(latents, locked_mask, phase, i)

    print("-" * 46)
    if locked_mask is not None:
        pct = 100 * locked_mask.mean()
        print(f"Locked mask: {locked_mask.sum()} / {locked_mask.size} tokens "
              f"({pct:.1f}%)")
    else:
        print("No mask locked (hard-fallback)")

    image = editor._decode(latents)

    if locked_mask is None:
        locked_mask = soft_prior_fallback_mask(prior_soft)

    editor.memory.add_layer(prompt, locked_mask,
                            token_span=s.obj_token_span,
                            object_phrase=object_phrase)
    editor._prev_latent = latents.detach()

    _save_summary_grid(records, editor.ph, editor.pw, out_dir)
    return image, locked_mask


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from diffusers import FluxPipeline
    from flux_seqedit.pipeline import SequentialEditor, EditConfig

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    collect = set(args.collect_blocks) if args.collect_blocks else None
    cfg = EditConfig(
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        collect_blocks=collect,
    )
    editor = SequentialEditor(pipe, cfg)

    # If a prior image is supplied, encode it and register as the committed background
    # so the BCG path is exercised exactly as it would be in a real second edit.
    if args.prior_image:
        from PIL import Image as PILImage
        prior_img = PILImage.open(args.prior_image).convert("RGB")

        # Re-encode through VAE to get the clean packed latent that _prev_latent holds.
        img_tensor = pipe.image_processor.preprocess(prior_img)
        img_tensor = img_tensor.to(pipe.vae.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            raw = pipe.vae.encode(img_tensor).latent_dist.sample()
        raw = (raw - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        packed, _ = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels // 4,
            args.height, args.width,
            raw.dtype, raw.device,
            generator=None,
        )
        # Use the re-encoded tensor, not a random latent.
        # (prepare_latents returns random noise; we replace it with our encoded image)
        import torch.nn.functional as F
        h_lat = args.height // pipe.vae_scale_factor
        w_lat = args.width  // pipe.vae_scale_factor
        # Unpack VAE output channels to match packed latent dimensions.
        # FLUX packs 2×2 spatial blocks → channel dimension × 4.
        raw2 = raw.reshape(1, raw.shape[1], h_lat // 2, 2, w_lat // 2, 2)
        raw2 = raw2.permute(0, 2, 4, 1, 3, 5)
        raw2 = raw2.reshape(1, (h_lat // 2) * (w_lat // 2), raw.shape[1] * 4)
        editor._prev_latent = raw2.to(pipe.transformer.device)

        print(f"Prior image loaded from {args.prior_image}  "
              f"(prev_latent shape: {editor._prev_latent.shape})")

    image, mask = run_diagnostic_edit(
        editor, args.prompt, args.object_phrase, args.seed, args.out_dir,
    )

    out_img = os.path.join(args.out_dir, "output.png")
    image.save(out_img)
    print(f"\nOutput image → {out_img}")

    if mask is not None:
        np.save(os.path.join(args.out_dir, "mask.npy"), mask)


if __name__ == "__main__":
    main()
