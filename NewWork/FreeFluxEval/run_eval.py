"""
FreeFlux downstream evaluation runner.

Tests FreeFlux non-rigid attention injection across 100 prompts spanning
10 task categories that go beyond the paper's three target tasks:
  color_change, texture_change, weather_change, time_of_day, style_transfer,
  object_replacement, attribute_change, action_change, seasonal_change, lighting_change

Uses run_non_rigid.py's pipeline and generation function — only adds category
filtering, progress reporting, resume support, and a summary JSON.

Usage
-----
# Run all 100
python NewWork/FreeFluxEval/run_eval.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/freeflux_downstream_eval.json \\
    --device cuda --cpu_offload --save_images

# Run one category only
python NewWork/FreeFluxEval/run_eval.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/freeflux_downstream_eval.json \\
    --category color_change \\
    --device cuda --cpu_offload --save_images

# Quick smoke-test: first 5 prompts
python NewWork/FreeFluxEval/run_eval.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/freeflux_downstream_eval.json \\
    --limit 5 \\
    --device cuda --cpu_offload --save_images

# Resume an interrupted run (skip already-saved outputs)
python NewWork/FreeFluxEval/run_eval.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/freeflux_downstream_eval.json \\
    --resume --device cuda --cpu_offload --save_images
"""

import argparse
import json
import os
import sys
import time

# Repo root is three levels up from NewWork/FreeFluxEval/
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Reproduce.FreeFlux.non_rigid.run_non_rigid import (
    load_freeflux_pipeline,
    run_non_rigid_edit,
)

_ALL_CATEGORIES = [
    "color_change",
    "texture_change",
    "weather_change",
    "time_of_day",
    "style_transfer",
    "object_replacement",
    "attribute_change",
    "action_change",
    "seasonal_change",
    "lighting_change",
]


def load_config(config_path: str) -> list:
    with open(config_path) as f:
        data = json.load(f)
    global_defaults = data.get("global", {})
    runs = []
    for run in data["runs"]:
        merged = {**global_defaults, **run}
        if "name" not in merged:
            raise ValueError(f"Each run must have a 'name' field: {run}")
        runs.append(merged)
    return runs


def _fmt_duration(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_eval(args):
    runs = load_config(args.config)

    # Filter by category
    if args.category:
        if args.category not in _ALL_CATEGORIES:
            print(f"Unknown category '{args.category}'. Valid: {_ALL_CATEGORIES}")
            sys.exit(1)
        runs = [r for r in runs if r.get("category") == args.category]
        print(f"Filtered to category '{args.category}': {len(runs)} runs")

    # Apply limit
    if args.limit:
        runs = runs[:args.limit]
        print(f"Limiting to first {args.limit} runs")

    # Resume: skip runs whose output directory already has both images
    if args.resume:
        pending = []
        skipped = 0
        for r in runs:
            run_dir = os.path.join(args.out_dir, r["name"])
            if (os.path.exists(os.path.join(run_dir, "source.png")) and
                    os.path.exists(os.path.join(run_dir, "edited.png"))):
                skipped += 1
            else:
                pending.append(r)
        print(f"Resume mode: skipping {skipped} already-done run(s), {len(pending)} remaining")
        runs = pending

    if not runs:
        print("No runs to execute.")
        return

    print(f"\n[FreeFlux downstream eval] Loading model from {args.model_path} ...")
    pipe = load_freeflux_pipeline(
        args.model_path, args.hf_token,
        device=args.device, cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print("[FreeFlux downstream eval] Model loaded.\n")

    summary = []
    durations = []
    total = len(runs)
    eval_start = time.time()

    for i, cfg in enumerate(runs, start=1):
        name           = cfg["name"]
        category       = cfg.get("category", "unknown")
        source_prompt  = cfg["source_prompt"]
        target_prompt  = cfg["target_prompt"]
        n_steps        = cfg.get("n_steps", 28)
        guidance_scale = cfg.get("guidance_scale", 3.5)
        height         = cfg.get("height", 1024)
        width          = cfg.get("width", 1024)
        max_seq_len    = cfg.get("max_sequence_length", 512)
        seed           = cfg.get("seed", 42)

        eta_str = ""
        if durations:
            avg = sum(durations) / len(durations)
            remaining = avg * (total - i + 1)
            eta_str = f"  ETA: {_fmt_duration(remaining)}"

        print(f"[{i:3d}/{total}] {name}  [{category}]{eta_str}")
        print(f"         src: {source_prompt}")
        print(f"         tgt: {target_prompt}")

        run_dir = os.path.join(args.out_dir, name)
        if args.save_images:
            os.makedirs(run_dir, exist_ok=True)

        t0 = time.time()
        status = "ok"
        error_msg = ""
        try:
            src_img, edited_img = run_non_rigid_edit(
                pipe,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                n_steps=n_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                max_sequence_length=max_seq_len,
                seed=seed,
            )
            if args.save_images:
                src_img.save(os.path.join(run_dir, "source.png"))
                edited_img.save(os.path.join(run_dir, "edited.png"))
                print(f"         saved → {run_dir}/")
        except Exception as e:
            status = "error"
            error_msg = str(e)
            print(f"         ERROR: {e}")

        elapsed = time.time() - t0
        durations.append(elapsed)
        print(f"         done in {_fmt_duration(elapsed)}\n")

        summary.append({
            "name":          name,
            "category":      category,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "status":        status,
            "error":         error_msg,
            "duration_s":    round(elapsed, 1),
        })

    total_elapsed = time.time() - eval_start
    ok_count    = sum(1 for r in summary if r["status"] == "ok")
    error_count = sum(1 for r in summary if r["status"] == "error")

    print("=" * 60)
    print(f"[FreeFlux downstream eval] Complete")
    print(f"  Runs: {total}  |  OK: {ok_count}  |  Errors: {error_count}")
    print(f"  Total time: {_fmt_duration(total_elapsed)}")
    if durations:
        print(f"  Avg per run: {_fmt_duration(sum(durations)/len(durations))}")

    # Write summary JSON
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "ok": ok_count,
            "errors": error_count,
            "total_duration_s": round(total_elapsed, 1),
            "runs": summary,
        }, f, indent=2)
    print(f"  Summary → {summary_path}")

    if error_count:
        print(f"\n  Failed runs:")
        for r in summary:
            if r["status"] == "error":
                print(f"    {r['name']}: {r['error']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FreeFlux downstream task evaluation (100 prompts, 10 categories)"
    )
    parser.add_argument("--config", type=str,
                        default="prompts/freeflux_downstream_eval.json",
                        help="Path to the evaluation JSON config file")
    parser.add_argument("--category", type=str, default=None,
                        choices=_ALL_CATEGORIES,
                        help="Run only this category (default: all 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N prompts (useful for smoke testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output images already exist")
    parser.add_argument("--model_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--out_dir", type=str,
                        default="results/freeflux/downstream_eval",
                        help="Root output directory (one sub-folder per run)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save source.png and edited.png for each run")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
