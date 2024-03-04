from typing import Tuple

import torch
from torch import nn, Tensor

from models.backbones import ResNet
from models.segmodel import SegModel
from utils.blocks import ConvBN, ConvBNReLU, DualConvBNReLU, UpSample


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_conv = ConvBNReLU(in_channels, out_channels, 3, 1, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBN(out_channels, out_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.in_conv(feature)
        return torch.mul(feature, self.attention(feature))


class SpatialPath(nn.Module):
    """ (b, 64, h/2, w/2) ---> (b, 128, h/8, w/8) """

    def __init__(self) -> None:
        super().__init__()
        self.duo_cbr = DualConvBNReLU(64, 64, 3, 2, 1)
        self.out_conv = ConvBNReLU(64, 128, 1, 1, 0)

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.duo_cbr(feature)
        return self.out_conv(feature)


class FeatureFusionModule(nn.Module):
    """ (b, 128, h/8, w/8) * 2 ---> (b, 1, h/8, w/8) """

    def __init__(self) -> None:
        super().__init__()
        self.cbr = ConvBNReLU(256, 1, 1, 1, 0)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1, 1, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, context: Tensor, spatial: Tensor) -> Tensor:
        feature = torch.cat((context, spatial), dim=1)
        pred = self.cbr(feature)
        return pred + torch.mul(pred, self.attention(pred))


class Encoder(nn.Module):
    """
    (b, 3, h, w) ---> (b, 3, h, w),
                      (b, 128, h/8, w/8)
    """

    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet(18)
        self.conv = ConvBNReLU(512, 128, 3, 1, 1)
        self.arm1 = AttentionRefinementModule(256, 128)
        self.arm2 = AttentionRefinementModule(512, 128)
        self.up2 = UpSample(2)
        self.out_conv = ConvBNReLU(128 * 3, 128, 1, 1, 0)

    def forward(self, image: Tensor) -> Tuple[Tensor, ...]:
        feature1, feature4, feature5 = self.resnet(image)[0], self.resnet(image)[3], self.resnet(image)[4]
        conv_feature5 = self.conv(feature5)
        conv_feature5 = self.up2(conv_feature5)
        attention_feature5 = self.arm2(feature5)
        attention_feature5 = self.up2(attention_feature5)
        feature4 = self.arm1(feature4)
        context = self.out_conv(torch.cat((feature4, attention_feature5, conv_feature5), dim=1))
        return feature1, self.up2(context)


class Decoder(nn.Module):
    """ Get Pred """

    def __init__(self) -> None:
        super().__init__()
        self.spatial_path = SpatialPath()
        self.ffm = FeatureFusionModule()
        self.out_conv = nn.Sequential(
            UpSample(8),
            nn.Conv2d(1, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, features: Tuple[Tensor, Tensor]) -> Tensor:
        low_feature, context = features
        spatial = self.spatial_path(low_feature)
        pred = self.ffm(context, spatial)
        return self.out_conv(pred)


class BiSeNet(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), Decoder())
