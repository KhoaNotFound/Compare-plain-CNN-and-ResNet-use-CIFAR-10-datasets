import torch
import torch.nn as nn
import torch.nn.init as init

from .visualization import plot_activations

class ResNet(nn.Module):
    """Eight convolutions with three identity skips; independent of CNN.

    Layers 1–2 are the stem. Pairs 3–4, 5–6 and 7–8 each compute
    ReLU(F(x) + x). This small comparison model is not ResNet-18.
    """

    def __init__(self) -> None:
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(3 if index == 0 else 32, 32, kernel_size=3, padding=1, bias=True)
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

    def extra_repr(self) -> str:
        return "identity skips: input of Conv 3/5/7 + BN output of Conv 4/6/8, before ReLU"

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

        # Padding=1 preserves spatial dimensions for identity additions.
        relu_features = x
        for index, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            # Layers 1–2 form the stem; pairs 3–4, 5–6, 7–8 form blocks.
            if index in (2, 4, 6):
                identity = relu_features
            conv_features = conv(relu_features)
            normalized_features = batch_norm(conv_features)
            # The ONLY extra operation in ResNet: add the block input before ReLU.
            if index in (3, 5, 7):
                normalized_features = normalized_features + identity
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
