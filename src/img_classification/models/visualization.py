"""Shared activation plotting; contains no model layers or forward logic."""
import torch


def plot_activations(
    inputs: torch.Tensor,
    conv_features: torch.Tensor,
    relu_features: torch.Tensor,
    pooled_features: torch.Tensor,
    fc1_features: torch.Tensor,
    logits: torch.Tensor,
    model_name: str,
) -> None:
    """Plot the last convolution's feature maps and dense activations."""
    import matplotlib.pyplot as plt

    stages = (
        ("Conv2d 8", conv_features),
        ("ReLU after Conv2d 8", relu_features),
        ("MaxPool + AdaptiveAvgPool", pooled_features),
    )
    max_maps = 16
    dense_stages = (("FC1 + ReLU", fc1_features), ("FC2 logits", logits))
    fig = plt.figure(figsize=(12, 14))
    grid = fig.add_gridspec(
        1 + len(stages) + len(dense_stages),
        1,
        height_ratios=[4] + [3] * len(stages) + [1] * len(dense_stages),
    )

    input_ax = fig.add_subplot(grid[0, 0])
    input_image = inputs[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
    input_ax.imshow(input_image)
    input_ax.set_title("Original input image (first image in batch)")
    input_ax.axis("off")

    for row, (stage_name, activations) in enumerate(stages):
        feature_maps = activations[0].detach().cpu()
        shown_maps = min(feature_maps.shape[0], max_maps)
        stage_grid = grid[row + 1, 0].subgridspec(1, max_maps)
        for col in range(max_maps):
            ax = fig.add_subplot(stage_grid[0, col])
            ax.axis("off")
            if col < shown_maps:
                ax.imshow(feature_maps[col], cmap="viridis")
                if row == 0:
                    ax.set_title(f"Map {col}", fontsize=8)
        fig.text(0.01, 0.73 - row * 0.17, stage_name, rotation=90, va="center")

    for row_offset, (stage_name, activations) in enumerate(
        dense_stages, start=1 + len(stages)
    ):
        values = activations[0].detach().cpu().flatten()
        ax = fig.add_subplot(grid[row_offset, 0])
        ax.bar(range(values.numel()), values.numpy())
        ax.set_title(stage_name, loc="left", fontsize=9)
        ax.set_xticks(range(values.numel()))
        ax.set_xlabel("Neuron")

    fig.suptitle(f"{model_name} activations for the first image in the batch")
    fig.tight_layout(rect=(0.04, 0, 1, 0.98))


    plt.show()
