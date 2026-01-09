import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RESAAggregator(nn.Module):
    """
    Recurrent Feature-Shift Aggregator.

    Shifts feature maps in 4 directions (down, up, right, left) with
    geometrically decreasing shift amounts, allowing each pixel to
    aggregate information from distant locations.

    The shift amount at iteration i is: size // 2^(num_iters - i)
    For num_iters=4 and H=36: shifts are 2, 4, 9, 18

    Args:
        channels: Number of input/output channels
        num_iters: Number of shift iterations per direction
        kernel_size: Kernel size for convolutions
        alpha: Scaling factor for aggregated features
    """

    def __init__(
        self,
        channels: int = 128,
        num_iters: int = 4,
        kernel_size: int = 9,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()

        self.num_iters = num_iters
        self.alpha = alpha

        # Vertical shift convolutions: (1, kernel_size) for horizontal context
        self.convs_d = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, kernel_size),
                      padding=(0, kernel_size // 2), bias=False)
            for _ in range(num_iters)
        ])
        self.convs_u = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, kernel_size),
                      padding=(0, kernel_size // 2), bias=False)
            for _ in range(num_iters)
        ])

        # Horizontal shift convolutions: (kernel_size, 1) for vertical context
        self.convs_r = nn.ModuleList([
            nn.Conv2d(channels, channels, (kernel_size, 1),
                      padding=(kernel_size // 2, 0), bias=False)
            for _ in range(num_iters)
        ])
        self.convs_l = nn.ModuleList([
            nn.Conv2d(channels, channels, (kernel_size, 1),
                      padding=(kernel_size // 2, 0), bias=False)
            for _ in range(num_iters)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, C, H, W) with aggregated spatial context
        """
        x = x.clone()
        _, _, H, W = x.shape

        # Vertical: down then up
        for i in range(self.num_iters):
            shift = H // (2 ** (self.num_iters - i))
            idx_d = (torch.arange(H, device=x.device) + shift) % H
            x = x + self.alpha * F.relu(self.convs_d[i](x[:, :, idx_d, :]))

        for i in range(self.num_iters):
            shift = H // (2 ** (self.num_iters - i))
            idx_u = (torch.arange(H, device=x.device) - shift) % H
            x = x + self.alpha * F.relu(self.convs_u[i](x[:, :, idx_u, :]))

        # Horizontal: right then left
        for i in range(self.num_iters):
            shift = W // (2 ** (self.num_iters - i))
            idx_r = (torch.arange(W, device=x.device) + shift) % W
            x = x + self.alpha * F.relu(self.convs_r[i](x[:, :, :, idx_r]))

        for i in range(self.num_iters):
            shift = W // (2 ** (self.num_iters - i))
            idx_l = (torch.arange(W, device=x.device) - shift) % W
            x = x + self.alpha * F.relu(self.convs_l[i](x[:, :, :, idx_l]))

        return x
