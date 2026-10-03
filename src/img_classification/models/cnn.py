from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init

class CNN(nn.Module):
    """Eight Conv + ReLU layers followed by pooling and two Linear layers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.convs = nn.ModuleList([
            nn.Conv2d(3 if index == 0 else 32, 32, kernel_size=3, bias=True)
            for index in range(8)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 3, bias=True)
        self.fc2 = nn.Linear(3, 10, bias=True)
        self.relu = nn.ReLU()
        self.conv_weight_init()
        self.fc_weight_init()
        
    def conv_weight_init(self):
        for conv in self.convs:
            init.kaiming_uniform_(
                conv.weight,
                mode='fan_in',
                nonlinearity='relu'
            )
            if conv.bias is not None:
                init.zeros_(conv.bias)
        
    
    def fc_weight_init(self):
        init.kaiming_uniform_(
            self.fc1.weight,
            mode='fan_in',
            nonlinearity='relu',
        )
        init.xavier_uniform_(self.fc2.weight)

        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

    def feed_forward(self, x: torch.Tensor, visualize: bool = False) -> torch.Tensor:
        """Run one forward pass; optionally display activations for the first image."""
        if x.ndim != 4:
            raise ValueError(f"Expected input shaped [N, C, H, W], got {tuple(x.shape)}")

        # Eight unpadded 3x3 convolutions shrink 32x32 inputs to 16x16.
        relu_features = x
        for conv in self.convs:
            conv_features = conv(relu_features)
            relu_features = self.relu(conv_features)
        pooled_features = self.pool(relu_features)
        pooled_features = self.avgpool(pooled_features)
        flattened_features = torch.flatten(pooled_features, start_dim=1)

        fc1_features = self.fc1(flattened_features)
        fc1_activated = self.relu(fc1_features)
        logits = self.fc2(fc1_activated)

        if visualize:
            self._plot_activations(
                x,
                conv_features,
                relu_features,
                pooled_features,
                fc1_activated,
                logits,
            )

        return logits

    def forward(self, x: torch.Tensor, visualize: bool = False) -> torch.Tensor:
        return self.feed_forward(x, visualize=visualize)

    @staticmethod
    def _plot_activations(
        inputs: torch.Tensor,
        conv_features: torch.Tensor,
        relu_features: torch.Tensor,
        pooled_features: torch.Tensor,
        fc1_features: torch.Tensor,
        logits: torch.Tensor,
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

        fig.suptitle("CNN activations for the first image in the batch")
        fig.tight_layout(rect=(0.04, 0, 1, 0.98))
        
        
        
        plt.show()
