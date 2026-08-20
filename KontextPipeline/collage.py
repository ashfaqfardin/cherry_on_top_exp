from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    H, W   = img.shape[:2]
    small  = cv2.resize(img, (256, 256))
    mask_s = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    eroded = cv2.erode(mask_s, np.ones((5, 5), np.uint8), iterations=2)
    mask_s = eroded if eroded.any() else mask_s
    sx  = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0)
    mag = np.max(mag, axis=-1) * mask_s; mag[mag < thresh] = 0.0
    mag3 = np.stack([mag, mag, mag], axis=-1)
    edge = (mag3.astype(np.float32) / 255.0 * small.astype(np.float32)).astype(np.uint8)
    return cv2.resize(edge, (W, H))


def build_collage_scene(
    scene:        Image.Image,
    obj_img:      Image.Image,
    ref_mask:     np.ndarray,
    target_bbox:  Tuple[int, int, int, int],
    collage_mode: str = "full",
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Paste obj_img into scene at target_bbox.

    FLUX Kontext receives the collage as its reference image, so it sees both
    obj_img and scene in a single prior without needing a separate mask input.
    """
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    ys, xs = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using centre crop")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh // 4, 3 * oh // 4, ow // 4, 3 * ow // 4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop  = obj_np[oy1:oy2, ox1:ox2]
    mask_crop = ref_mask[oy1:oy2, ox1:ox2]

    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  {zone_w}×{zone_h} px")

    if collage_mode == "sobel":
        obj_z      = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z     = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    oh, ow  = obj_crop.shape[:2]
    scale   = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h   = max(4, int(oh * scale))
    new_w   = max(4, int(ow * scale))
    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = (cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h)) > 0.5).astype(np.float32)

    if collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8), mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:
        paste_layer = obj_rs

    mask_3 = np.stack([mask_rs] * 3, axis=-1)
    dy  = (zone_h - new_h) // 2
    dx  = (zone_w - new_w) // 2
    py1, px1 = max(0, y1 + dy), max(0, x1 + dx)
    py2, px2 = min(H, py1 + new_h), min(W, px1 + new_w)
    ch, cw   = py2 - py1, px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region zero — returning scene unchanged")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]; mask_3 = mask_3[:ch, :cw]
    collage_np  = scene_np.copy()
    collage_np[py1:py2, px1:px2] = (
        paste_layer * mask_3 + collage_np[py1:py2, px1:px2] * (1.0 - mask_3)
    )
    collage_np = np.clip(collage_np, 0, 255).astype(np.uint8)
    print(f"    [COLLAGE] Pasted {new_w}×{new_h} at [{py1}:{py2},{px1}:{px2}]  mode={collage_mode}")
    return Image.fromarray(collage_np), (y1, y2, x1, x2)


def build_removal_collage(
    scene:     Image.Image,
    mask_np:   np.ndarray,
    height:    int,
    width:     int,
    dilate_px: int = 8,
) -> Image.Image:
    """Inpaint the object region using Telea to get a clean background canvas."""
    scene_np     = np.array(scene.convert("RGB").resize((width, height), Image.LANCZOS))
    inpaint_mask = (mask_np > 127).astype(np.uint8)
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((k, k), np.uint8), iterations=1)
    inpainted = cv2.inpaint(scene_np, inpaint_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    print(f"    [REMOVAL] Telea inpaint: {int(inpaint_mask.sum())} px erased")
    return Image.fromarray(inpainted)


def _collage_obj_mask(collage: Image.Image, scene: Image.Image,
                      threshold: int = 10) -> np.ndarray:
    """Binary uint8 mask of pixels that differ between collage and scene."""
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask
