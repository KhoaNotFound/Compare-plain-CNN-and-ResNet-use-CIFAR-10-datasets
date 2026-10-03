"""Inspect a small sample from the locally saved CIFAR-10 dataset."""
#%%
import matplotlib.pyplot as plt
from datasets import load_from_disk

from img_classification.config import RAW_DIR
from img_classification.dataset import vectorize_dataset


def main() -> None:
    dataset = load_from_disk(str(RAW_DIR))
    raw_train_dataset = dataset["train"]
    classes = raw_train_dataset.features["label"].names
    # print(classes)
    train_dataset = vectorize_dataset(raw_train_dataset)
    sample = train_dataset.select(range(min(5, len(train_dataset))))[:]
    first_image = sample["image"][0]
    # first_image_R_channel = first_image[0][:][:]
    # print(first_image_R_channel)
    print(first_image)
    print(f"Image tensor shape: {tuple(first_image.shape)}")
    print(f"Image tensor dtype: {first_image.dtype}")
    print(f"Image tensor range: [{first_image.min().item():.3f}, {first_image.max().item():.3f}]")

    fig, axes = plt.subplots(1, len(sample["image"]), figsize=(12, 3), squeeze=False)
    for ax, image, label in zip(axes[0], sample["image"], sample["label"]):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(classes[label])
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# %%
