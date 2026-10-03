"""Train CIFAR-10, select on validation, and report held-out test predictions."""

import argparse
import random
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from img_classification.config import PROCESSED_DIR, load_train_config
from img_classification.artifacts import save_json, save_checkpoint
from img_classification.dataset import load_cached_dataset
from img_classification.models import CNN


def prepare_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Transfer compact uint8 images, then normalize on the training device."""
    return images.to(device, non_blocking=True).to(torch.float32).div_(255)


def train_one_epoch(
    model: CNN,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = prepare_batch(images, device)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; check data and learning rate")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

# def train_one_epoch(
#     model: CNN,
#     loader: DataLoader,
#     loss_fn: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     scaler: torch.amp.GradScaler | None = None,
# ) -> tuple[float, float]:
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_examples = 0

#     for images, labels in loader:
#         images = prepare_batch(images, device)
#         labels = labels.to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)
#         with torch.autocast(device_type=device.type, enabled=scaler is not None):
#             logits = model(images)
#             loss = loss_fn(logits, labels)

#         if not torch.isfinite(loss):
#             raise RuntimeError("Non-finite training loss; check data and learning rate")

#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)  # Đưa gradient về giá trị thực trước khi tính norm
#         else:
#             loss.backward()

#         # 1. Kiểm tra gradient norm của từng layer
#         grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

#         if not grad_norms:
#             print("[CẢNH BÁO]: Không có gradient nào được tính! Kiểm tra require_grad hoặc loss.backward().")
#         elif all(g == 0.0 for g in grad_norms):
#             print("[CẢNH BÁO]: Gradient bị triệt tiêu hoàn toàn về 0.0 (Dying ReLU hoặc Vanishing Gradient)!")
#         else:
#             print(f"Gradient norm min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}")

#         # 2. Kiểm tra xem scaler có skip optimizer step không (nếu dùng FP16)
#         if scaler is not None:
#             scale_before = scaler.get_scale()
#             scaler.step(optimizer)
#             scaler.update()
#             scale_after = scaler.get_scale()
#             if scale_after < scale_before:
#                 print("[CẢNH BÁO]: Gradient bị inf/NaN, scaler đã skip bước cập nhật optimizer!")
#         else:
#             optimizer.step()

#         batch_size = labels.size(0)
#         total_loss += loss.item() * batch_size
#         total_correct += (logits.argmax(dim=1) == labels).sum().item()
#         total_examples += batch_size

#     return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def evaluate(
    model: CNN,
    loader: DataLoader,
    device: torch.device,
    preview_count: int = 16,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    total_correct = 0
    total_examples = 0
    preview_images = []
    preview_labels = []
    preview_predictions = []

    for images, labels in loader:
        logits = model(prepare_batch(images, device))
        predictions = logits.argmax(dim=1).cpu()

        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

        remaining = preview_count - len(preview_labels)
        if remaining > 0:
            count = min(remaining, labels.size(0))
            preview_images.append(images[:count].to(torch.float32).div_(255))
            preview_labels.extend(labels[:count].tolist())
            preview_predictions.extend(predictions[:count].tolist())

    accuracy = total_correct / total_examples
    return (
        accuracy,
        torch.cat(preview_images) if preview_images else torch.empty((0, 3, 32, 32)),
        torch.tensor(preview_labels),
        torch.tensor(preview_predictions),
    )


def plot_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    count = len(images)
    columns = 4
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)

    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue

        image = images[index].permute(1, 2, 0).clamp(0, 1)
        true_name = class_names[labels[index].item()]
        predicted_name = class_names[predictions[index].item()]
        correct = labels[index] == predictions[index]

        ax.imshow(image)
        ax.set_title(
            f"True: {true_name}\nPred: {predicted_name}",
            color="green" if correct else "red",
        )
        ax.axis("off")

    fig.suptitle("CIFAR-10 test predictions (green = correct, red = incorrect)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10; select weights using validation only.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        config = load_train_config(args.config, epochs=args.epochs, batch_size=args.batch_size,
                                   lr=args.lr, seed=args.seed, val_fraction=args.val_fraction, amp=args.amp)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA GPU is available.")
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(selected_device)
    output = args.output_dir or Path("outputs/local") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("checkpoint.pt", "best.pt", "metrics.json", "config.json")):
        parser.error(f"Run already exists: {output}. Choose a fresh --output-dir.")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset, class_names = load_cached_dataset(PROCESSED_DIR / "train.pt")
    test_dataset, test_class_names = load_cached_dataset(PROCESSED_DIR / "test.pt")
    if class_names != test_class_names:
        raise ValueError("Train/test class names differ. Rebuild caches.")
    if len(dataset) < 2:
        raise ValueError("Need at least two training examples for a validation split")
    val_size = max(1, min(len(dataset) - 1, int(len(dataset) * config.val_fraction)))
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )
    loader_options = dict(batch_size=config.batch_size, num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed), **loader_options)
    val_loader = DataLoader(val_dataset, **loader_options)
    test_loader = DataLoader(test_dataset, **loader_options)
    model = CNN().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda") if config.amp and device.type == "cuda" else None
    save_json(config.to_dict(), output / "config.json")
    save_json({"train_indices": train_dataset.indices, "validation_indices": val_dataset.indices},
              output / "split.json")
    print(f"Device: {device}; AMP: {scaler is not None}; artifacts: {output}", flush=True)
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        val_accuracy, _, _, _ = evaluate(model, val_loader, device, preview_count=0)
        if not math.isfinite(train_loss):
            raise RuntimeError("Non-finite epoch loss")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "validation_accuracy": val_accuracy})
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                      "epoch": epoch, "class_names": class_names, "config": config.to_dict()}
        save_checkpoint(checkpoint, output / "checkpoint.pt")
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            save_checkpoint(checkpoint, output / "best.pt")
        save_json({"history": history, "best_epoch": best_epoch,
                   "best_validation_accuracy": best_accuracy}, output / "metrics.json")
        print(f"Epoch {epoch}/{config.epochs}: loss={train_loss:.4f} "
              f"train={train_accuracy:.2%} validation={val_accuracy:.2%}", flush=True)

    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_accuracy, images, labels, predictions = evaluate(model, test_loader, device)
    save_json({"history": history, "best_epoch": best_epoch,
               "best_validation_accuracy": best_accuracy, "test_accuracy": test_accuracy,
               "device": str(device), "amp_enabled": scaler is not None,
               "config": config.to_dict(), "torch_version": str(torch.__version__)}, output / "metrics.json")
    plot_predictions(images, labels, predictions, class_names, output / "predictions.png")
    print(f"Best epoch: {best_epoch}; test accuracy: {test_accuracy:.2%}", flush=True)


if __name__ == "__main__":
    main()
