"""Download CIFAR-10 if needed and cache decoded uint8 tensors on disk."""

import argparse

import torch
from datasets import load_dataset, load_from_disk

from img_classification.config import PROCESSED_DIR, RAW_DIR
from img_classification.dataset import image_to_uint8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild existing tensor caches.")
    args = parser.parse_args()
    if not args.rebuild and all((PROCESSED_DIR / f"{split}.pt").is_file() for split in ("train", "test")):
        print(f"Using existing caches: {PROCESSED_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / "dataset_dict.json").is_file():
        dataset = load_from_disk(str(RAW_DIR))
    else:
        dataset = load_dataset("uoft-cs/cifar10")
        dataset.save_to_disk(str(RAW_DIR))

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"{split}.pt"
        if path.is_file() and not args.rebuild:
            print(f"Using existing cache: {path}")
            continue

        source = dataset[split]
        images = torch.empty((len(source), 3, 32, 32), dtype=torch.uint8)
        labels = torch.empty(len(source), dtype=torch.long)
        for index, example in enumerate(source):
            images[index].copy_(image_to_uint8(example["img"]))
            labels[index] = int(example["label"])

        # Replace only after saving completes, avoiding partially written caches.
        temporary_path = path.with_suffix(".pt.tmp")
        torch.save(
            {"images": images, "labels": labels, "class_names": source.features["label"].names},
            temporary_path,
        )
        temporary_path.replace(path)
        print(f"Saved {split}: {len(source):,} images, {path.stat().st_size / 1024**2:.1f} MiB → {path}")


if __name__ == "__main__":
    main()
