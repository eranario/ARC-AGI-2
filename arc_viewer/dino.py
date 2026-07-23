"""DINOv3 image embeddings for ARC grid renderings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError as exc:
        raise ImportError(
            "DINOv3 embeddings require torch. Install with: "
            "uv sync --extra dino"
        ) from exc


def load_dinov3(model_name: str = DEFAULT_DINO_MODEL, device: str | None = None):
    """Load DINOv3 processor + model. Requires transformers>=4.56 and HF access."""
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError(
            "DINOv3 embeddings require transformers>=4.56. "
            "Install with: uv sync --extra dino"
        ) from exc

    device = _resolve_device(device)
    print(f"Loading DINOv3 model {model_name} on {device}...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError as exc:
        raise SystemExit(
            f"Failed to load `{model_name}`.\n"
            "DINOv3 weights are gated on Hugging Face. Accept the license at\n"
            f"  https://huggingface.co/{model_name}\n"
            "then authenticate (`huggingface-cli login` or set HF_TOKEN) and retry."
        ) from exc

    model.eval()
    model.to(device)
    return processor, model, device


def embed_images(
    images: Sequence[Image.Image],
    *,
    model_name: str = DEFAULT_DINO_MODEL,
    batch_size: int = 32,
    device: str | None = None,
    processor=None,
    model=None,
) -> np.ndarray:
    """Embed RGB PIL images with DINOv3 pooled/CLS features."""
    import torch

    if processor is None or model is None:
        processor, model, device = load_dinov3(model_name, device=device)
    else:
        device = device or next(model.parameters()).device

    if not images:
        return np.zeros((0, 0), dtype=np.float32)

    chunks: list[np.ndarray] = []
    total = len(images)
    for start in range(0, total, batch_size):
        batch = list(images[start : start + batch_size])
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0]
        chunks.append(feats.detach().float().cpu().numpy())
        done = min(start + batch_size, total)
        if done == total or done % (batch_size * 10) < batch_size:
            print(f"  DINOv3 embedded {done}/{total}")

    return np.concatenate(chunks, axis=0)


def embed_image_paths(
    paths: Iterable[str | Path],
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper that opens image paths then embeds them."""
    images = [Image.open(Path(p)).convert("RGB") for p in paths]
    return embed_images(images, **kwargs)
