from typing import Tuple

import torch
from torch import nn, Tensor

from models.backbones import ResNet
from models.segmodel import SegModel
from utils.blocks import DualConvBNReLU, UpSample


class Encoder(nn.Module):
    """ Get Features """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet(34)

    def forward(self, image: Tensor) -> Tuple[Tensor, ...]:
        return self.backbone(image)


class Decoder(nn.Module):
    """ Get Pred """

    def __init__(self) -> None:
        super().__init__()
        self.up2 = UpSample(2)
        self.duo_cbr1 = DualConvBNReLU(256 + 512, 256, 3, 1, 1)
        self.duo_cbr2 = DualConvBNReLU(128 + 256, 128, 3, 1, 1)
        self.duo_cbr3 = DualConvBNReLU(64 + 128, 64, 3, 1, 1)
        self.duo_cbr4 = DualConvBNReLU(128, 64, 3, 1, 1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        temp = self.up2(features[4])
        temp = self.duo_cbr1(torch.cat((features[3], temp), dim=1))
        temp = self.up2(temp)
        temp = self.duo_cbr2(torch.cat((features[2], temp), dim=1))
        temp = self.up2(temp)
        temp = self.duo_cbr3(torch.cat((features[1], temp), dim=1))
        temp = self.up2(temp)
        temp = self.duo_cbr4(torch.cat((features[0], temp), dim=1))
        temp = self.up2(temp)
        return self.out_conv(temp)


class UNet(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), Decoder())
