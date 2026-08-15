from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
LETTERBOX_PAD_RGB = (124, 116, 104)


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image array, got shape={array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def letterbox_rgb_tensor(
    image: np.ndarray,
    *,
    image_size: int,
    pad_rgb: tuple[int, int, int] = LETTERBOX_PAD_RGB,
) -> torch.Tensor:
    array = ensure_uint8_image(image)
    pil = Image.fromarray(array, mode="RGB")
    pil.thumbnail((int(image_size), int(image_size)), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (int(image_size), int(image_size)), color=pad_rgb)
    offset = ((int(image_size) - pil.width) // 2, (int(image_size) - pil.height) // 2)
    canvas.paste(pil, offset)
    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()


def stack_letterboxed_images(images: Iterable[np.ndarray], *, image_size: int) -> torch.Tensor:
    tensors = [letterbox_rgb_tensor(image, image_size=image_size) for image in images]
    if not tensors:
        raise ValueError("Expected at least one image to stack.")
    return torch.stack(tensors, dim=0)

