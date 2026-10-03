#%%
from img_classification.model import CNN
from img_classification.dataset import vectorize_dataset

"""Inspect a small sample from the locally saved CIFAR-10 dataset."""
#%%
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk

from img_classification.config import RAW_DIR


def main() -> None:
    dataset = load_from_disk(str(RAW_DIR))
    raw_train_dataset = dataset["train"]
    classes = raw_train_dataset.features["label"].names
    # print(classes)
    train_dataset = vectorize_dataset(raw_train_dataset)
    sample = train_dataset.select(range(min(5, len(train_dataset))))[:]
    batch_image = torch.stack(sample["image"])

    original_fig, original_ax = plt.subplots(figsize=(4, 4))
    original_ax.imshow(batch_image[0].permute(1, 2, 0).numpy())
    original_ax.set_title("Original first image in batch")
    original_ax.axis("off")

    print(type(batch_image))
    print(batch_image.shape)
    model = CNN()
    with torch.no_grad():
        logits = model(batch_image, visualize=True)
    # first_image_R_channel = batch_image[0][:][:]
    # print(first_image_R_channel)
    print("Batch shape:", tuple(batch_image.shape))
    print("Logits shape:", tuple(logits.shape))
    print(logits.tolist())

if __name__ == "__main__":
    main()

# %%
