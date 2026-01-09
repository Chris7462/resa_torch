import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ExistHead(nn.Module):
    """
    Lane existence prediction head.

    Predicts whether each lane exists in the image.

    Architecture:
        Dropout → Conv(in_channels→num_classes, 1x1) → Softmax →
        AdaptiveAvgPool(1,1) → Flatten →
        FC(num_classes→hidden) → ReLU →
        FC(hidden→num_lanes)

    Args:
        in_channels: Number of input channels from aggregator
        num_classes: Number of segmentation classes (including background)
        hidden_channels: Hidden layer size in FC layers

    Output:
        Existence logits of shape (B, num_lanes) where num_lanes = num_classes - 1
        Use BCEWithLogitsLoss for training.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 5,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()

        num_lanes = num_classes - 1

        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_classes, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_lanes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W) from aggregator

        Returns:
            Existence logits of shape (B, num_lanes)
        """
        x = self.dropout(x)
        x = self.conv(x)
        x = F.softmax(x, dim=1)
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
