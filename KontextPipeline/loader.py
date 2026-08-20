from __future__ import annotations

import torch
from diffusers import FluxKontextPipeline


def load_sam2(
    model_id: str = "facebook/sam2-hiera-small",
    cache_dir: str = "./models",
    device: str = "cuda",
):
    """Load SAM2 image predictor for zero-shot segmentation."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor.from_pretrained(model_id, cache_dir=cache_dir)
        predictor.model.to(device)
        print(f"  SAM2 loaded: {model_id}")
        return predictor
    except ImportError:
        raise ImportError(
            "SAM2 not installed. Run: pip install 'git+https://github.com/facebookresearch/sam2.git'"
        )


def load_kontext_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
    hf_token: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "./models",
    cpu_offload: bool = False,
) -> FluxKontextPipeline:
    pipe = FluxKontextPipeline.from_pretrained(
        model_path, torch_dtype=dtype, token=hf_token, cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_vlm(model_id: str, cache_dir: str, device: str = "cpu"):
    """
    Load a vision-language model for placement anchor prediction.

    device='cpu'  — bfloat16 on CPU. Safe alongside FLUX on any GPU.
    device='cuda' — tries 4-bit NF4 (bitsandbytes) first, falls back to bf16.
    """
    from transformers import AutoProcessor

    print(f"  Loading VLM '{model_id}' on {device} ...")

    def _try_load(load_kwargs, to_device):
        is_q25 = "Qwen2.5" in model_id or "Qwen2_5" in model_id
        classes = []
        if is_q25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        try:
            from transformers import Qwen2VLForConditionalGeneration
            classes.append(Qwen2VLForConditionalGeneration)
        except ImportError:
            pass
        if not is_q25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        last_err = None
        for cls in classes:
            try:
                m = cls.from_pretrained(model_id, **load_kwargs)
                return (m.to(device) if to_device else m).eval()
            except Exception as e:
                last_err = e
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
            return (m.to(device) if to_device else m).eval()
        except Exception as e:
            last_err = e
        raise RuntimeError(f"Could not load VLM '{model_id}'. Last error: {last_err}")

    model = None
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            model = _try_load(
                dict(quantization_config=quant, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            print("  VLM loaded (4-bit NF4 GPU)")
        except Exception as e:
            print(f"    [VLM] 4-bit load failed ({e!s:.80}) — falling back to bf16")
        if model is None:
            model = _try_load(
                dict(torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            print("  VLM loaded (bf16 GPU)")
    else:
        model = _try_load(
            dict(torch_dtype=torch.bfloat16, cache_dir=cache_dir),
            to_device=True,
        )
        print("  VLM loaded (bf16 CPU)")

    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, processor
