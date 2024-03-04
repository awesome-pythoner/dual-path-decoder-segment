from typing import Tuple

import torch.nn.functional as F
from torch import nn, Tensor


class DilatedConvBN(nn.Module):
    """ Dilated Conv + BN """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int] = 1,
            stride: int | Tuple[int, int] = 1,
            padding: str | int | Tuple[int, int] = 0,
            dilation: int | Tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.db = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.db(x)


class ConvBN(nn.Module):
    """ Conv + BN """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int] = 1,
            stride: int | Tuple[int, int] = 1,
            padding: str | int | Tuple[int, int] = 0
    ) -> None:
        super().__init__()
        self.cb = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1, bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.cb(x)


class DilatedConvBNReLU(nn.Module):
    """ Dilated Conv + BN + ReLU """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int] = 1,
            stride: int | Tuple[int, int] = 1,
            padding: str | int | Tuple[int, int] = 0,
            dilation: int | Tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.dbr = nn.Sequential(
            DilatedConvBN(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.ReLU(True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.dbr(x)


class ConvBNReLU(nn.Module):
    """ Conv + BN + ReLU """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int] = 1,
            stride: int | Tuple[int, int] = 1,
            padding: str | int | Tuple[int, int] = 0
    ) -> None:
        super().__init__()
        self.cbr = DilatedConvBNReLU(in_channels, out_channels, kernel_size, stride, padding, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.cbr(x)


class DualConvBNReLU(nn.Module):
    """ CBR * 2 """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int] = 1,
            stride: int | Tuple[int, int] = 1,
            padding: str | int | Tuple[int, int] = 0
    ) -> None:
        super().__init__()
        self.double_cbr = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding),
            ConvBNReLU(out_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_cbr(x)


class UpSample(nn.Module):
    def __init__(self, ratio: int = 2) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, size=(x.size(2) * self.ratio, x.size(3) * self.ratio), mode="bilinear",
                             align_corners=True)


class Shortcut(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int | Tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.shortcut = ConvBN(in_channels, out_channels, 1, stride, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x)


class ResBlock(nn.Module):
    """ Generate ResBlocks """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int | Tuple[int, int] = 1,
            deeper_then_34: bool = False
    ) -> None:
        super().__init__()
        self.plain = PlainBlock(in_channels, out_channels, stride, deeper_then_34)
        self.shortcut = Shortcut(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        x = self.plain(x) + self.shortcut(x)
        return nn.ReLU(True)(x)


class PlainBlock(nn.Module):
    """ Generate ResBlocks w/o ShortCut """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int | Tuple[int, int] = 1,
            deeper_then_34: bool = False
    ) -> None:
        super().__init__()
        if not deeper_then_34:
            self.block = _PlainBlock1(in_channels, out_channels, stride)
        else:
            self.block = _PlainBlock2(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _PlainBlock1(nn.Module):
    """ When Layers <= 34 """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int | Tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, stride, 1),
            ConvBN(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _PlainBlock2(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int | Tuple[int, int] = 1
    ) -> None:
        super().__init__()
        mid_channels = in_channels // (4 if stride == 1 else 2)
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, 1, stride, 0),
            ConvBNReLU(mid_channels, mid_channels, 3, 1, 1),
            ConvBN(mid_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DilatedBottleNeck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dilation: int | Tuple[int, int] = 2
    ) -> None:
        super().__init__()
        mid_channels = in_channels // (4 if in_channels == out_channels else 2)
        self.block = nn.Sequential(
            DilatedConvBNReLU(in_channels, mid_channels, 1, 1, 0, dilation),
            DilatedConvBNReLU(mid_channels, mid_channels, 3, 1, dilation, dilation),
            DilatedConvBN(mid_channels, out_channels, 1, 1, 0, dilation)
        )
        self.shortcut = Shortcut(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x) + self.shortcut(x)
        return nn.ReLU(True)(x)
