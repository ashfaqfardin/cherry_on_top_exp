from __future__ import annotations

import torch
from PIL import Image

LORA_ID      = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."


def generate_from_sketch(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int, lora_id: str, device: str,
) -> Image.Image:
    sketch_pil = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    print(f"  Loading LoRA: {lora_id}")
    pipe.load_lora_weights(lora_id)
    prompt = (
        f"{LORA_TRIGGER} {description} on a plain white background, "
        "photorealistic, no shadows, studio lighting, high quality."
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    obj_img = pipe(
        image=sketch_pil, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width,
        max_sequence_length=512,
        generator=generator, output_type="pil",
    ).images[0]
    pipe.unload_lora_weights()
    return obj_img
