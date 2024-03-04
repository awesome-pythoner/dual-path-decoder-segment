import torch
from torch import nn, Tensor


class HalfBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        return 0.5 * self.criterion(pred, label)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-8
        # self.smooth = 0

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        loss = (-self.alpha * (1 - pred).pow(self.gamma) * label * torch.log(pred + self.smooth) -
                (1 - self.alpha) * pred.pow(self.gamma) * (1 - label) * torch.log(1 - pred + self.smooth))
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1.) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.flatten = nn.Flatten()

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        pred, label = self.flatten(pred), self.flatten(label)
        # label.pow(n) == label
        intersection = (2 * pred * label).sum()
        union = pred.pow(2).sum() + label.sum()
        return 1. - (intersection + self.epsilon) / (union + self.epsilon)
