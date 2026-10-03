"""Convert the saved Hugging Face image dataset into PyTorch tensors."""
from typing import Any
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import TensorDataset

def image_to_uint8(image: Image.Image) -> torch.Tensor:
    """Decode an image into a CHW uint8 tensor without normalization."""
    pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1)


def vectorize_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor with shape (C, H, W) in [0, 1]."""
    return image_to_uint8(image).to(torch.float32).div_(255)


def load_cached_dataset(path: Path) -> tuple[TensorDataset, list[str]]:
    """Load decoded uint8 images and labels into CPU RAM once."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing tensor cache: {path}. Run python -m img_classification.prepare_data first."
        )
    cache = torch.load(path, map_location="cpu", weights_only=True)
    images, labels = cache["images"], cache["labels"]
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError(f"Invalid CIFAR-10 image cache: {path}")
    if labels.dtype != torch.long or labels.shape != (len(images),):
        raise ValueError(f"Invalid CIFAR-10 label cache: {path}")
    names = cache["class_names"]
    if not isinstance(names, list) or len(names) != 10 or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Invalid CIFAR-10 class names: {path}")
    if len(images) == 0 or labels.min() < 0 or labels.max() >= len(names):
        raise ValueError(f"Empty cache or out-of-range labels: {path}")
    return TensorDataset(images, labels), names

def _vectorize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": [vectorize_image(image) for image in batch["img"]],
        "label": batch["label"],
    }


def vectorize_dataset(dataset: Dataset) -> Dataset:
    """Return a dataset that emits vectorized images when examples are read."""
    return dataset.with_transform(_vectorize_batch)
