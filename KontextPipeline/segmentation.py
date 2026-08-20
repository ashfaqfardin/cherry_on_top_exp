"""
segmentation.py — SAM2 zero-shot segmentation.

Replaces all legacy mask generation: color thresholding, morphological operations,
and pixel-diff. SAM2 receives the VLM bounding box as a prompt and returns
a clean binary mask with accurate object boundaries.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


class SAM2Segmenter:
    """
    Zero-shot segmentation using SAM2 with bounding-box prompts.
    Input:  image + (x1, y1, x2, y2) pixel bbox from VLMGrounder
    Output: binary mask (H, W) uint8 with 1 = object, 0 = background
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def segment(
        self,
        image: Image.Image,
        bbox_pixels: Tuple[int, int, int, int],
        multimask: bool = True,
    ) -> np.ndarray:
        """
        Run SAM2 with a box prompt.
        Returns best-scoring binary mask (H, W) uint8.
        """
        img_np = np.array(image.convert("RGB"))
        self.predictor.set_image(img_np)

        x1, y1, x2, y2 = bbox_pixels
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=multimask,
        )

        # Pick highest-confidence mask
        best  = int(np.argmax(scores))
        mask  = masks[best].astype(np.uint8)
        print(f"    [SAM2] Coverage: {mask.mean()*100:.1f}%  "
              f"confidence: {scores[best]:.3f}  "
              f"(chose mask {best+1}/{len(masks)})")
        return mask

    def segment_token_mask(
        self,
        image: Image.Image,
        bbox_pixels: Tuple[int, int, int, int],
        h_tok: int,
        w_tok: int,
    ) -> np.ndarray:
        """
        Segment and resize to FLUX token grid (h_tok, w_tok).
        Returns float32 mask in [0, 1], shape (h_tok * w_tok,).
        """
        import cv2
        mask_px = self.segment(image, bbox_pixels)
        mask_tok = cv2.resize(
            mask_px.astype(np.float32), (w_tok, h_tok),
            interpolation=cv2.INTER_AREA,
        )
        return mask_tok.reshape(-1)
