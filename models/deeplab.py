from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.backbones import DilatedResNet50
from models.segmodel import SegModel
from utils.blocks import ConvBNReLU, DilatedConvBNReLU, DualConvBNReLU, UpSample


class Encoder(nn.Module):
    """ Get Features """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DilatedResNet50()
        self.aspp = AtrousSpatialPyramidPooling()

    def forward(self, image: Tensor) -> Tuple[Tensor, ...]:
        conv_features = self.backbone(image)
        aspp_feature = self.aspp(conv_features[-1])
        return *conv_features, aspp_feature


class AtrousSpatialPyramidPooling(nn.Module):
    """ (b, 2048, h/16, w/16) ---> (b, 256, h/16, w/16) """

    def __init__(self) -> None:
        super().__init__()
        self.parallel1 = ConvBNReLU(2048, 256, 1, 1, 0)
        self.parallel2 = DilatedConvBNReLU(2048, 256, 3, 1, 6, 6)
        self.parallel3 = DilatedConvBNReLU(2048, 256, 3, 1, 12, 12)
        self.parallel4 = DilatedConvBNReLU(2048, 256, 3, 1, 18, 18)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(2048, 256, 1, 1, 0)
        )
        self.out_conv = ConvBNReLU(1280, 256, 1, 1, 0)

    def forward(self, feature: Tensor) -> Tensor:
        feature1 = self.parallel1(feature)
        feature2 = self.parallel2(feature)
        feature3 = self.parallel3(feature)
        feature4 = self.parallel4(feature)
        feature5 = self.avg_pool(feature)
        feature5 = F.interpolate(feature5, size=(feature.size(2), feature.size(3)), mode='bilinear',
                                 align_corners=True)
        feature = torch.cat((feature1, feature2, feature3, feature4, feature5), dim=1)
        return self.out_conv(feature)


class DetailPath(nn.Module):
    """ (b, 256, h/4, w/4), (b, 256, h/16, w/16) ---> (b, 128, h/4, w/4) """

    def __init__(self) -> None:
        super().__init__()
        self.in_conv = ConvBNReLU(256, 48, 1, 1, 0)
        self.up4 = UpSample(4)
        self.duo_cbr = DualConvBNReLU(256 + 48, 256, 3, 1, 1)

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        detail, aspp_feature = features[1], features[-1]
        detail = self.in_conv(detail)
        aspp_feature = self.up4(aspp_feature)
        detail_fusion = torch.cat((aspp_feature, detail), dim=1)
        return self.duo_cbr(detail_fusion)


class DeepLabV3PlusHead(nn.Module):
    """ (b, 128, h/4, w/4) ---> (b, 1, h, w) """

    def __init__(self) -> None:
        super().__init__()
        self.up4 = UpSample(4)
        self.out_conv = nn.Sequential(
            nn.Conv2d(256, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, *preds: Tensor) -> Tensor:
        pred = torch.cat(preds, dim=1)
        pred = self.up4(pred)
        return self.out_conv(pred)


class DeepLabV3PlusDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.path = DetailPath()
        self.head = DeepLabV3PlusHead()

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        fusion = self.path(features)
        return self.head(fusion)


class DeepLabV3Decoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.duo_conv = DualConvBNReLU(256, 64, 3, 1, 1)
        self.up16 = UpSample(16)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, features: Tensor) -> Tensor:
        feature = self.duo_conv(features[-1])
        feature = self.up16(feature)
        return self.out_conv(feature)


class DeepLabV3Plus(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), DeepLabV3PlusDecoder())


class DeepLabV3(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), DeepLabV3Decoder())
