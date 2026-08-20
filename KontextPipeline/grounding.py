"""
grounding.py — VLM-based spatial placement prediction.

Replaces the 20-step DAAM heatmap pipeline with a single VLM forward pass
that outputs a JSON bounding box directly in normalized [0, 1] coordinates.
"""

from __future__ import annotations

import json
import re
from typing import Tuple

import torch
from PIL import Image


_PROMPT_TMPL = (
    "Look at this room image. Where would a {description} be placed naturally on the floor? "
    "Give a small, tight bounding box (NOT the full image). "
    "The box should be at most 40% of the image width and 40% of the image height. "
    "Focus on the visible floor area. "
    'Output ONLY valid JSON: {{"xmin": float, "ymin": float, "xmax": float, "ymax": float}} '
    "where 0.0=top-left, 1.0=bottom-right. No explanation."
)


class VLMGrounder:
    """
    Single-shot VLM spatial grounding.
    Outputs normalized (xmin, ymin, xmax, ymax) placement bbox for an object.
    """

    def __init__(self, model, processor):
        self.model     = model
        self.processor = processor

    @torch.no_grad()
    def predict_bbox(self, scene: Image.Image, description: str) -> dict:
        """
        Returns {"xmin": float, "ymin": float, "xmax": float, "ymax": float}
        all in [0, 1] normalized coordinates.
        """
        query = _PROMPT_TMPL.format(description=description)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": scene.convert("RGB")},
                {"type": "text",  "text": query},
            ],
        }]

        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[scene.convert("RGB")],
            padding=True, return_tensors="pt"
        ).to(next(self.model.parameters()).device)

        out_ids = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        answer  = self.processor.decode(
            out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        print(f"    [GND] Raw VLM output: {answer[:120]}")

        # Extract JSON from response (may have surrounding text)
        match = re.search(r'\{[^}]+\}', answer, re.DOTALL)
        if match:
            try:
                raw = json.loads(match.group())
                bbox = {
                    "xmin": float(raw.get("xmin", raw.get("x_min", 0.2))),
                    "ymin": float(raw.get("ymin", raw.get("y_min", 0.5))),
                    "xmax": float(raw.get("xmax", raw.get("x_max", 0.8))),
                    "ymax": float(raw.get("ymax", raw.get("y_max", 0.9))),
                }
                # Clamp and validate
                for k in ("xmin", "ymin", "xmax", "ymax"):
                    bbox[k] = max(0.0, min(1.0, bbox[k]))
                if bbox["xmax"] <= bbox["xmin"]:
                    bbox["xmax"] = min(1.0, bbox["xmin"] + 0.35)
                if bbox["ymax"] <= bbox["ymin"]:
                    bbox["ymax"] = min(1.0, bbox["ymin"] + 0.35)
                # Reject degenerate full-image bbox (model not grounding properly)
                area = (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"])
                if area > 0.50:
                    print(f"    [GND] Rejected degenerate bbox (area={area:.2f} > 0.50) → fallback")
                else:
                    print(f"    [GND] Parsed bbox: {bbox}  area={area:.2f}")
                    return bbox
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"    [GND] JSON parse failed: {e}")

        # Floor-centre: lower-middle third — a safe default for most room objects
        print(f"    [GND] Fallback: floor-centre region")
        return {"xmin": 0.30, "ymin": 0.58, "xmax": 0.70, "ymax": 0.88}

    def to_pixels(self, bbox: dict, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bbox → integer pixel coords (x1, y1, x2, y2)."""
        x1 = int(bbox["xmin"] * width)
        y1 = int(bbox["ymin"] * height)
        x2 = int(bbox["xmax"] * width)
        y2 = int(bbox["ymax"] * height)
        # Enforce minimum size
        if x2 - x1 < 64:  x2 = min(width,  x1 + 64)
        if y2 - y1 < 64:  y2 = min(height, y1 + 64)
        return x1, y1, x2, y2
