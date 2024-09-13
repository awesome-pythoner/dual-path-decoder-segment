from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.backbones import DilatedResNet50
from utils.blocks import ConvBNReLU, DilatedConvBNReLU, DualConvBNReLU, UpSample


class ASPP(nn.Module):
    """ (b, 2048, h/16, w/16) ---> (b, 256, h/16, w/16) """

    def __init__(self) -> None:
        super().__init__()
        self.parallel1 = ConvBNReLU(2048, 256, 1, 1, 0)
        self.parallel2 = DilatedConvBNReLU(2048, 256, 3, 1, 2, 2)
        self.parallel3 = DilatedConvBNReLU(2048, 256, 3, 1, 4, 4)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(2048, 256, 1, 1, 0)
        )
        self.out_conv = ConvBNReLU(1024, 256, 1, 1, 0)

    def forward(self, feature: Tensor) -> Tensor:
        feature1 = self.parallel1(feature)
        feature2 = self.parallel2(feature)
        feature3 = self.parallel3(feature)
        feature4 = self.avg_pool(feature)
        feature4 = F.interpolate(feature4, size=(feature.size(2), feature.size(3)), mode='bilinear',
                                 align_corners=True)
        feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)
        return self.out_conv(feature)


class Encoder(nn.Module):
    """ Get Features """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DilatedResNet50()
        self.aspp = ASPP()

    def forward(self, image: Tensor) -> Tuple[Tensor, ...]:
        conv_features = self.backbone(image)
        aspp_feature = self.aspp(conv_features[-1])
        return *conv_features, aspp_feature


class Decoder(nn.Module):
    """ Get Pred """

    def __init__(self, *paths: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.paths = nn.ModuleList(paths)
        self.head = head

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        fusions = [path(features) for path in self.paths]
        return self.head(*fusions)

    def get_paths(self) -> nn.ModuleList:
        return self.paths


class SegModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image: Tensor) -> Tensor:
        features = self.encoder(image)
        return self.decoder(features)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_paths(self) -> nn.Module:
        return self.decoder.get_paths()

    def load_weights(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())


class DetailPath(nn.Module):
    """ (b, 256, h/4, w/4), (b, 256, h/16, w/16) ---> (b, 128, h/4, w/4) """

    def __init__(self) -> None:
        super().__init__()
        self.in_conv = ConvBNReLU(256, 64, 1, 1, 0)
        self.up4 = UpSample(4)
        self.dual_cbr = DualConvBNReLU(256 + 64, 128, 3, 1, 1)

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        detail, aspp_feature = features[1], features[-1]
        detail = self.in_conv(detail)
        aspp_feature = self.up4(aspp_feature)
        detail_fusion = torch.cat((aspp_feature, detail), dim=1)
        return self.dual_cbr(detail_fusion)


class ContextPath(nn.Module):
    """ (b, 512, h/8, w/8), (b, 256, h/16, w/16) ---> (b, 128, h/4, w/4) """

    def __init__(self) -> None:
        super().__init__()
        self.in_conv = ConvBNReLU(512, 64, 1, 1, 0)
        self.up2 = UpSample(2)
        self.dual_cbr = DualConvBNReLU(256 + 64, 128, 3, 1, 1)

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        context, aspp_feature = features[2], features[-1]
        context = self.in_conv(context)
        aspp_feature = self.up2(aspp_feature)
        context_fusion = torch.cat((aspp_feature, context), dim=1)
        context_fusion = self.dual_cbr(context_fusion)
        return self.up2(context_fusion)


class Head(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.up4 = UpSample(4)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, *preds: Tensor) -> Tensor:
        pred = torch.cat(preds, dim=1)
        pred = self.up4(pred)
        return self.out_conv(pred)


class SingleHead(Head):
    """ (b, 128, h/4, w/4) ---> (b, 1, h, w) """

    def __init__(self) -> None:
        super().__init__(128)


class DualPathHead(Head):
    """ (b, 128, h/4, w/4) * 2 ---> (b, 1, h, w) """

    def __init__(self) -> None:
        super().__init__(256)
        self.dual_cbr = DualConvBNReLU(256, 64, 3, 1, 1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, *preds: Tensor) -> Tensor:
        pred = torch.cat(preds, dim=1)
        pred = self.dual_cbr(pred)
        pred = self.up4(pred)
        return self.out_conv(pred)


class DetailDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__(DetailPath(), head=SingleHead())


class ContextDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__(ContextPath(), head=SingleHead())


class DualPathDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__(DetailPath(), ContextPath(), head=DualPathHead())


class DualPathDecoderSeg(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), DualPathDecoder())


class DetailDecoderSeg(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), DetailDecoder())

    def load_weights(self, model: DualPathDecoderSeg) -> None:
        self.encoder.load_state_dict(model.get_encoder().state_dict())
        self.get_paths()[0].load_state_dict(model.get_paths()[0].state_dict())


class ContextDecoderSeg(SegModel):
    def __init__(self) -> None:
        super().__init__(Encoder(), ContextDecoder())

    def load_weights(self, model: DualPathDecoderSeg) -> None:
        self.encoder.load_state_dict(model.get_encoder().state_dict())
        self.get_paths()[0].load_state_dict(model.get_paths()[1].state_dict())


class DualPathDecoderSsp(DualPathDecoderSeg):
    pass
