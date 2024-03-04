from typing import List

from torch import nn, Tensor

from utils.blocks import ConvBNReLU, ResBlock, DilatedBottleNeck

channels_dict = {
    False: (64, 64, 128, 256, 512),
    True: (64, 256, 512, 1024, 2048)
}
num_blocks_dict = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3)
}


class ResNet(nn.Module):
    """
    (b, 3, h, w) ---> (b, 64, h/2, w/2),
                      (b, 64 or 256, h/4, w/4),\n
                      (b, 128 or 512, h/8, w/8),\n
                      (b, 256 or 1024, h/16, w/16),\n
                      (b, 512 or 2048, h/32, w/32)
    """

    def __init__(self, depth: int = 34) -> None:
        super().__init__()
        assert depth in (18, 34, 50, 101, 152), "Depth Must be 18, 34, 50, 101 or 152!"
        self.depth = depth
        channels = channels_dict[depth > 34]

        self.in_conv = ConvBNReLU(3, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet = [self._make_layer(
            channels[i],
            channels[i + 1],
            num_blocks_dict[depth][i],
            1 if i == 0 else 2
        ) for i in range(4)]
        self.resnet = nn.Sequential(*self.resnet)

    def _make_layer(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            stride: int
    ) -> nn.Module:
        layers = [ResBlock(in_channels, out_channels, stride, self.depth > 34)]
        for _ in range(num_blocks - 1):
            layers.append(ResBlock(out_channels, out_channels, 1, self.depth > 34))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.in_conv(x)
        y = [x]
        x = self.maxpool(x)
        for i in range(4):
            x = self.resnet[i](x)
            y.append(x)
        return y


class DilatedResNet50(ResNet):
    """
    (b, 3, h, w) ---> (b, 64, h/2, w/2),
                      (b, 256, h/4, w/4),\n
                      (b, 512, h/8, w/8),\n
                      (b, 1024, h/16, w/16),\n
                      (b, 2048, h/16, w/16)
    """

    def __init__(self) -> None:
        super().__init__(50)
        channels = channels_dict[True]

        self.dilated_resnet = [self._make_layer(
            channels[i],
            channels[i + 1],
            num_blocks_dict[50][i],
            1 if i == 0 else 2
        ) for i in range(3)]
        self.dilated_resnet.append(
            self._make_dilated_layer(1024, 2048, 3, 2)
        )
        self.resnet = nn.Sequential(*self.dilated_resnet)

    @staticmethod
    def _make_dilated_layer(
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            dilation: int = 2
    ) -> nn.Module:
        layers = [DilatedBottleNeck(in_channels, out_channels, dilation=dilation)]
        for _ in range(num_blocks - 1):
            layers.append(DilatedBottleNeck(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)
