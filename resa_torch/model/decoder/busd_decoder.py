import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .registry import DECODERS


class NonBottleneck1D(nn.Module):
    """
    Non-bottleneck-1D block with factorized convolutions.

    Architecture:
        Conv(3x1) → ReLU → Conv(1x3) → BN → ReLU →
        Conv(3x1, dilated) → ReLU → Conv(1x3, dilated) → BN → Dropout →
        Residual add → ReLU

    Args:
        channels: Number of input/output channels
        dropout: Dropout probability
        dilation: Dilation rate for second conv pair
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3)

        self.conv3x1_2 = nn.Conv2d(channels, channels, (3, 1),
                                   padding=(dilation, 0), dilation=(dilation, 1), bias=True)
        self.conv1x3_2 = nn.Conv2d(channels, channels, (1, 3),
                                   padding=(0, dilation), dilation=(1, dilation), bias=True)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.conv3x1_1(x))
        out = F.relu(self.bn1(self.conv1x3_1(out)))

        out = F.relu(self.conv3x1_2(out))
        out = self.bn2(self.conv1x3_2(out))

        if self.dropout.p > 0:
            out = self.dropout(out)

        return F.relu(out + x)


class UpsamplerBlock(nn.Module):
    """
    Upsampler block with transposed conv and skip connection.

    Architecture:
        Main path: ConvTranspose(2x) → BN → ReLU → NonBottleneck1D × 2
        Skip path: Conv(1x1) → BN → ReLU → Bilinear upsample
        Output: Main + Skip

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        # Main path: transposed conv with 2x upsample
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3,
            stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

        self.blocks = nn.Sequential(
            NonBottleneck1D(out_channels, dropout=0, dilation=1),
            NonBottleneck1D(out_channels, dropout=0, dilation=1),
        )

        # Skip path: 1x1 conv + bilinear upsample
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip_bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        # Main path
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)

        # Skip path (upsample to match main path output size)
        skip = F.relu(self.skip_bn(self.skip_conv(x)))
        skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)

        return out + skip


@DECODERS.register
class BUSDDecoder(nn.Module):
    """
    Bilateral Up-Sampling Decoder.

    3-stage upsampling decoder with skip connections.

    Architecture:
        UpsamplerBlock(128→64, 2x) →
        UpsamplerBlock(64→32, 2x) →
        UpsamplerBlock(32→16, 2x) →
        Conv(16→num_classes, 1x1)

    Total upsample: 8x

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes (including background)
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 7,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            UpsamplerBlock(in_channels, 64),
            UpsamplerBlock(64, 32),
            UpsamplerBlock(32, 16),
        )

        self.output_conv = nn.Conv2d(16, num_classes, kernel_size=1, bias=False)

    def forward(self, x: Tensor, output_size: tuple[int, int] = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            output_size: Target output size (H, W). If None, uses 8x input size.

        Returns:
            Segmentation logits of shape (B, num_classes, output_H, output_W)
        """
        x = self.layers(x)
        x = self.output_conv(x)

        # Adjust to exact output size if needed
        if output_size is not None and (x.shape[2] != output_size[0] or x.shape[3] != output_size[1]):
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)

        return x
