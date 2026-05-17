"""
Experiment 2 — Plot semantic sensitivity heatmap.

Reads results/semantic_sensitivity.npy and produces:
  results/semantic_heatmap.png          — standalone heatmap
  results/vitality_semantic_combined.png — vitality bars + heatmap side-by-side

Usage
-----
python experiments/plot_semantic_heatmap.py
    [--sensitivity results/semantic_sensitivity.npy]
    [--vitality    results/vitality_scores.json]
    [--threshold   0.92]
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

CATEGORIES     = ["colour", "style", "material", "texture", "shape", "layout", "object"]
PAPER_MM_VITAL = [0, 1, 17, 18]
PAPER_SIN_VITAL= [9, 34, 35, 37, 6]
N_MM           = 19
N_SINGLE       = 38


def plot_heatmap(matrix: np.ndarray, out_path: str,
                 vital_rows: list, threshold: float):
    """Draw the 57×7 sensitivity heatmap with vital layer markers."""
    layer_labels = (
        [f"MM-{i}" for i in range(N_MM)] +
        [f"S-{i}"  for i in range(N_SINGLE)]
    )

    fig, ax = plt.subplots(figsize=(11, 20))

    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            xticklabels=CATEGORIES,
            yticklabels=layer_labels,
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.3,
            linecolor="white",
            annot=False,
            cbar_kws={"label": "Mean |ΔCLIP distance| when bypassed"},
        )
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, fontsize=9)
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels, fontsize=7)
        plt.colorbar(im, ax=ax, label="Mean |ΔCLIP distance| when bypassed")

    # Mark vital layers with a red border on each row
    for row in vital_rows:
        rect = mpatches.FancyBboxPatch(
            (-0.5, row - 0.5), len(CATEGORIES), 1.0,
            linewidth=1.8, edgecolor="red", facecolor="none",
            transform=ax.transData, clip_on=False,
        )
        ax.add_patch(rect)

    ax.axhline(N_MM, color="black", linewidth=2, label="MM-DiT / single-stream boundary")
    ax.set_xlabel("Semantic category", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Layer Semantic Sensitivity\n"
                 "(change in CLIP inter-pair distance when block bypassed)\n"
                 "Red borders = paper's vital layers",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def plot_combined(matrix: np.ndarray, vitality_path: str,
                  out_path: str, threshold: float):
    """Side-by-side: vitality bars (left) + semantic heatmap (right)."""
    with open(vitality_path) as f:
        vdata = json.load(f)

    mm_means  = [float(np.mean(vdata["mm"][str(i)]))     for i in range(N_MM)]
    sin_means = [float(np.mean(vdata["single"][str(i)])) for i in range(N_SINGLE)]

    fig = plt.figure(figsize=(22, 20))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.05)

    # --- Left: vitality bars ---
    ax_v = fig.add_subplot(gs[0])
    all_means   = mm_means + sin_means
    vital_global = PAPER_MM_VITAL + [N_MM + i for i in PAPER_SIN_VITAL]
    colors = ["#d62728" if i in vital_global else "#1f77b4"
              for i in range(N_MM + N_SINGLE)]
    ax_v.barh(range(N_MM + N_SINGLE), all_means, color=colors, height=0.7)
    ax_v.axvline(threshold, color="red", linestyle="--", linewidth=1.2)
    ax_v.axhline(N_MM - 0.5, color="black", linewidth=2)
    ax_v.set_yticks(range(N_MM + N_SINGLE))
    ax_v.set_yticklabels(
        [f"MM-{i}" for i in range(N_MM)] + [f"S-{i}" for i in range(N_SINGLE)],
        fontsize=7
    )
    ax_v.invert_yaxis()
    ax_v.set_xlabel("DINOv2 similarity (↑ = less vital)", fontsize=9)
    ax_v.set_title("Layer Vitality\n(★ = paper vital layers)", fontsize=10)
    for i in vital_global:
        ax_v.text(all_means[i] + 0.002, i, "★", va="center", fontsize=8, color="#d62728")

    # --- Right: heatmap ---
    ax_h = fig.add_subplot(gs[1])
    layer_labels = (
        [f"MM-{i}" for i in range(N_MM)] +
        [f"S-{i}"  for i in range(N_SINGLE)]
    )
    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            xticklabels=CATEGORIES,
            yticklabels=[],
            cmap="YlOrRd",
            ax=ax_h,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Mean |ΔCLIP| when bypassed"},
        )
    else:
        im = ax_h.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax_h.set_xticks(range(len(CATEGORIES)))
        ax_h.set_xticklabels(CATEGORIES, fontsize=9)
        ax_h.set_yticks([])
        plt.colorbar(im, ax=ax_h, label="Mean |ΔCLIP| when bypassed")

    for row in vital_global:
        rect = mpatches.FancyBboxPatch(
            (-0.5, row - 0.5), len(CATEGORIES), 1.0,
            linewidth=1.8, edgecolor="red", facecolor="none",
            transform=ax_h.transData, clip_on=False,
        )
        ax_h.add_patch(rect)
    ax_h.axhline(N_MM - 0.5, color="black", linewidth=2)
    ax_h.set_xlabel("Semantic category", fontsize=9)
    ax_h.set_title("Semantic Sensitivity by Layer\n(red border = vital layer)", fontsize=10)

    fig.suptitle("FLUX Layer Analysis — Vitality × Semantic Specialisation",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", type=str,
                        default="results/semantic_sensitivity.npy")
    parser.add_argument("--vitality",    type=str,
                        default="results/vitality_scores.json")
    parser.add_argument("--threshold",   type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    matrix = np.load(args.sensitivity)

    vital_rows = PAPER_MM_VITAL + [N_MM + i for i in PAPER_SIN_VITAL]

    plot_heatmap(matrix, "results/semantic_heatmap.png", vital_rows, args.threshold)

    if os.path.exists(args.vitality):
        plot_combined(matrix, args.vitality,
                      "results/vitality_semantic_combined.png", args.threshold)
    else:
        print(f"No vitality scores at {args.vitality}, skipping combined plot.")
