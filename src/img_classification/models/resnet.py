"""Small residual counterpart of the eight-convolution CNN (not ResNet-18)."""
from .cnn import CNN


class ResNet(CNN):
    """Same padded convolutions and head; add three identity skip connections.

    Each block computes F(x) = BN(Conv(ReLU(BN(Conv(x))))),
    then ReLU(F(x) + x). Identity adds no parameters.
    """

    def __init__(self) -> None:
        super().__init__(padding=1, residual=True)
