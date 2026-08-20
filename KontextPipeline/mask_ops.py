from __future__ import annotations

import numpy as np
from PIL import Image


def _compute_obj_mask(obj_img: Image.Image,
                      grey: tuple = (128, 128, 128),
                      tolerance: int = 20) -> np.ndarray:
    """Binary uint8 mask (H, W): 1=object pixel, 0=background."""
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


def _neutralize_white_bg(img: Image.Image, threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    arr[white] = fill
    return Image.fromarray(arr)


def _obj_token_mask(obj_img: Image.Image, h_lat: int, w_lat: int,
                    grey: tuple = (128, 128, 128), tol: int = 10,
                    min_frac: float = 0.05) -> np.ndarray:
    """Bool mask (h_lat*w_lat,): True = object token."""
    arr = np.array(obj_img.convert("RGB").resize((w_lat, h_lat), Image.LANCZOS))
    is_grey  = (np.abs(arr[:,:,0].astype(int) - grey[0]) <= tol) & \
               (np.abs(arr[:,:,1].astype(int) - grey[1]) <= tol) & \
               (np.abs(arr[:,:,2].astype(int) - grey[2]) <= tol)
    is_white = (arr[:,:,0] >= 230) & (arr[:,:,1] >= 230) & (arr[:,:,2] >= 230)
    mask = ~(is_grey | is_white)
    if mask.mean() < min_frac:
        mask[:] = False
        mask[h_lat//4:3*h_lat//4, w_lat//4:3*w_lat//4] = True
    return mask.reshape(-1)


def _rect_mask_from_bbox(x1: int, y1: int, x2: int, y2: int,
                          height: int, width: int) -> np.ndarray:
    """Rectangle uint8 mask (H×W), 255 inside bbox."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _token_zone_from_mask_np(mask_np: np.ndarray, height: int, width: int,
                               pipe) -> np.ndarray:
    """Convert a uint8 pixel mask (H,W) to a flat bool token mask (n_gen,)."""
    vae_sf   = getattr(pipe, "vae_scale_factor", 8)
    h_lat    = height // (vae_sf * 2)
    w_lat    = width  // (vae_sf * 2)
    mask_pil = Image.fromarray(mask_np).resize((w_lat, h_lat), Image.NEAREST)
    zone     = (np.array(mask_pil) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_lat * w_lat} tokens ({100*zone.mean():.1f}%)")
    return zone
