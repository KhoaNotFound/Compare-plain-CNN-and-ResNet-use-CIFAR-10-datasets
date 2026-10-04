from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init

from .visualization import plot_activations

class CNN(nn.Module):
    """Eight Conv + ReLU layers followed by pooling and two Linear layers."""

    def __init__(self, *args: Any, padding: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.convs = nn.ModuleList([
            nn.Conv2d(3 if index == 0 else 32, 32, kernel_size=3, padding=padding, bias=True)
            for index in range(8)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(32) for _ in range(8)
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

        # Legacy padding=0 shrinks to 16x16; benchmark padding=1 keeps 32x32.
        relu_features = x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv_features = conv(relu_features)
            normalized_features = batch_norm(conv_features)
            relu_features = self.relu(normalized_features)
        pooled_features = self.pool(relu_features)
        pooled_features = self.avgpool(pooled_features)
        flattened_features = torch.flatten(pooled_features, start_dim=1)

        fc1_features = self.fc1(flattened_features)
        fc1_activated = self.relu(fc1_features)
        logits = self.fc2(fc1_activated)

        if visualize:
            plot_activations(
                x,
                conv_features,
                relu_features,
                pooled_features,
                fc1_activated,
                logits,
                model_name=type(self).__name__,
            )

        return logits

    def forward(self, x: torch.Tensor, visualize: bool = False) -> torch.Tensor:
        return self.feed_forward(x, visualize=visualize)
