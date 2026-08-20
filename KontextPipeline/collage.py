"""
collage.py — Minimal image compositing for V2.

For insertion: paste obj_img at VLM-predicted bbox to build the collage
reference image used in Pass B of flow injection.

For removal: fill the object region with smooth surrounding content so FLUX
has a plausible background reference to reconstruct from.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def paste_object(
    scene: Image.Image,
    obj_img: Image.Image,
    bbox_pixels: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Resize obj_img to fit the VLM bbox and paste it onto scene.
    Returns the composite (obj over scene at bbox).
    """
    x1, y1, x2, y2 = bbox_pixels
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    obj_resized = obj_img.convert("RGBA").resize((bw, bh), Image.LANCZOS)
    result = scene.copy().convert("RGBA")
    result.paste(obj_resized, (x1, y1), mask=obj_resized.split()[3])
    return result.convert("RGB")


def build_removal_reference(
    scene: Image.Image,
    sam2_mask: np.ndarray,   # (H, W) uint8  0/1
    blur_radius: int = 51,
) -> Image.Image:
    """
    Build a plausible background reference for FLUX by filling the
    object region with blurred surrounding content.

    Replaces Telea inpainting: Gaussian blur gives FLUX a smooth, low-frequency
    background hint. FLUX fills in the high-frequency details itself.
    """
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    mask_f   = sam2_mask.astype(np.float32)[:, :, None]  # (H, W, 1)

    # Blurred version provides the fill color for the masked region
    k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    blurred  = cv2.GaussianBlur(scene_np, (k, k), k // 3)

    # Slightly dilate mask so hard edges are fully covered
    dil_mask = cv2.dilate(sam2_mask, np.ones((15, 15), np.uint8), iterations=2)
    dil_f    = dil_mask.astype(np.float32)[:, :, None]

    filled = scene_np * (1.0 - dil_f) + blurred * dil_f
    return Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8))
