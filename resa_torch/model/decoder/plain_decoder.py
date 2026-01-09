import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .registry import DECODERS


@DECODERS.register
class PlainDecoder(nn.Module):
    """
    Simple decoder with dropout and bilinear upsampling.

    Architecture:
        Dropout → Conv(in_channels→num_classes, 1x1) → Bilinear upsample

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes (including background)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            output_size: Target output size (H, W)

        Returns:
            Segmentation logits of shape (B, num_classes, output_H, output_W)
        """
        x = self.dropout(x)
        x = self.conv(x)
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)

        return x
