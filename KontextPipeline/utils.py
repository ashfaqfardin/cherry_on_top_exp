from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def run_standard(
    pipe, canvas: Image.Image, prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    bcg_callback=None,
) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    cb_kwargs: dict = {}
    if bcg_callback is not None:
        cb_kwargs["callback_on_step_end"] = bcg_callback
        cb_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(
        prompt=prompt, image=canvas,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, generator=generator,
        **cb_kwargs,
    ).images[0]


def save_grid(images: List[Image.Image], titles: List[str], path: str,
              ncols: Optional[int] = None, figsize_per_cell=(4, 4)):
    n     = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
