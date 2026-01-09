import torch.nn as nn
from torch import Tensor
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
)


RESNET_CONFIGS = {
    'resnet18': (resnet18, ResNet18_Weights.DEFAULT, 512),
    'resnet34': (resnet34, ResNet34_Weights.DEFAULT, 512),
    'resnet50': (resnet50, ResNet50_Weights.DEFAULT, 2048),
    'resnet101': (resnet101, ResNet101_Weights.DEFAULT, 2048),
    'resnet152': (resnet152, ResNet152_Weights.DEFAULT, 2048),
}


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for RESA.

    Modifications from standard ResNet:
    1. Layer 3 and 4 use dilated convolutions instead of stride
    2. Final pooling and FC layers are removed
    3. Output conv reduces channels to specified output channels

    Output stride is 8 instead of 32.

    Args:
        arch: ResNet architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        pretrained: Whether to load pretrained ImageNet weights
        out_channels: Number of output channels (default: 128)
    """

    def __init__(
        self,
        arch: str = 'resnet34',
        pretrained: bool = True,
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        if arch not in RESNET_CONFIGS:
            raise ValueError(f"Unknown architecture: {arch}. Choose from {list(RESNET_CONFIGS.keys())}")

        model_fn, weights, in_channels = RESNET_CONFIGS[arch]

        # Load model with dilated convolutions in layer3 and layer4
        resnet = model_fn(
            weights=weights if pretrained else None,
            replace_stride_with_dilation=[False, True, True],
        )

        # Extract backbone layers (remove avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Output conv to reduce channels
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, out_channels, H/8, W/8)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.out_conv(x)

        return x
