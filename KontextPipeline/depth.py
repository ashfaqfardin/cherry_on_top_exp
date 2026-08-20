from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def load_depth_model(
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    cache_dir: str = "./models",
):
    """
    Load a monocular depth estimator.

    Returns (model, processor, kind_str).
    Falls back through: Depth-Anything-V2 → Intel/dpt-large → None (geometric).
    """
    candidates = [
        (model_id, "depth-anything"),
        ("depth-anything/Depth-Anything-V2-Large-hf", "depth-anything"),
    ]
    for mid, kind in candidates:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc  = AutoImageProcessor.from_pretrained(mid, cache_dir=cache_dir)
            model = AutoModelForDepthEstimation.from_pretrained(mid, cache_dir=cache_dir)
            print(f"  [Depth] Loaded '{mid}'")
            return model.eval(), proc, kind
        except Exception as e:
            print(f"  [Depth] '{mid}' unavailable: {e!s:.70}")
    print("  [Depth] No depth model loaded — using geometric floor fallback.")
    return None, None, "fallback"


@torch.no_grad()
def estimate_depth(model, processor, scene: Image.Image) -> Optional[np.ndarray]:
    """
    Run depth estimation on scene.
    Returns (H, W) float32 depth map normalised to [0, 1].
    Larger values = closer to camera (Depth Anything inverted-disparity convention).
    Returns None if model is None.
    """
    if model is None:
        return None
    inputs = processor(images=scene, return_tensors="pt")
    out = model(**inputs)
    depth = out.predicted_depth[0].numpy().astype(np.float32)
    lo, hi = depth.min(), depth.max()
    return (depth - lo) / (hi - lo + 1e-8)


def _depth_to_scene_size(depth_np: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.array(
        Image.fromarray((depth_np * 255).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0


def find_floor_bbox(
    depth_np:   Optional[np.ndarray],
    height:     int,
    width:      int,
    floor_frac: float = 0.42,
    pad_frac:   float = 0.06,
) -> Tuple[int, int, int, int]:
    """
    Select a placement bbox on the floor using depth-guided flat-region detection.

    1. Focus on lower (1 - floor_frac) of the image.
    2. Compute Sobel gradient of depth — low gradient = flat surface = floor.
    3. Threshold at 55th percentile → flat-surface binary mask.
    4. Largest connected component → bbox with inset padding.

    Geometric fallback (no depth model): lower-centre rectangle.
    """
    floor_y0 = int(height * floor_frac)

    if depth_np is None:
        print(f"    [PLACE] Geometric fallback: center-bottom region")
        return width // 5, floor_y0, 4 * width // 5, int(height * 0.90)

    depth = _depth_to_scene_size(depth_np, height, width)
    floor = depth[floor_y0:]

    sx = cv2.Sobel(floor, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(floor, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sx ** 2 + sy ** 2)

    flat_mask = (grad < np.percentile(grad, 55)).astype(np.uint8)
    k = np.ones((11, 11), np.uint8)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, k)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_mask)
    if n_labels <= 1:
        x1, y1 = width // 5, floor_y0 + (height - floor_y0) // 4
        x2, y2 = 4 * width // 5, int(height * 0.90)
        print(f"    [PLACE] Uniform floor — centre-bottom: x=[{x1},{x2}] y=[{y1},{y2}]")
        return x1, y1, x2, y2

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp    = (labels == largest)
    ys, xs  = np.where(comp)

    ry1, ry2 = int(ys.min()) + floor_y0, int(ys.max()) + floor_y0
    rx1, rx2 = int(xs.min()), int(xs.max())

    px = max(12, int((rx2 - rx1) * pad_frac))
    py = max(12, int((ry2 - ry1) * pad_frac))
    x1 = max(0,      rx1 + px)
    y1 = max(0,      ry1 + py)
    x2 = min(width,  rx2 - px)
    y2 = min(height, ry2 - py)

    area_frac = comp.sum() / (flat_mask.shape[0] * flat_mask.shape[1])
    print(f"    [PLACE] Floor bbox: x=[{x1},{x2}] y=[{y1},{y2}]  flat_area={area_frac*100:.1f}%")
    return x1, y1, x2, y2


def detect_object_on_floor(
    depth_np:   Optional[np.ndarray],
    height:     int,
    width:      int,
    floor_frac: float = 0.42,
) -> Tuple[int, int, int, int]:
    """
    Locate the most prominent non-floor object in the lower image region via
    depth anomaly detection. Used for removal when insertion bbox wasn't tracked.
    """
    floor_y0 = int(height * floor_frac)

    if depth_np is None:
        print(f"    [DETECT] Geometric fallback: centre-bottom")
        return width // 4, floor_y0, 3 * width // 4, int(height * 0.88)

    depth = _depth_to_scene_size(depth_np, height, width)
    floor = depth[floor_y0:]

    floor_med   = np.median(floor)
    floor_std   = floor.std()
    bump_thresh = floor_med + max(0.06, 0.8 * floor_std)
    obj_mask    = (floor > bump_thresh).astype(np.uint8)

    k = np.ones((9, 9), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask)
    if n_labels <= 1:
        print(f"    [DETECT] No clear depth anomaly — using centre-bottom fallback")
        return width // 4, floor_y0, 3 * width // 4, int(height * 0.88)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp    = (labels == largest)
    ys, xs  = np.where(comp)

    x1 = max(0,      int(xs.min()))
    y1 = max(0,      int(ys.min()) + floor_y0)
    x2 = min(width,  int(xs.max()))
    y2 = min(height, int(ys.max()) + floor_y0)

    print(f"    [DETECT] Depth-anomaly bbox: x=[{x1},{x2}] y=[{y1},{y2}]")
    return x1, y1, x2, y2
