from typing import Dict

import torch
from torch import Tensor

from utils.dataset import pred2mask


class ConfusionMatrix:
    def __init__(self):
        self.total = 0  # Total Pixels
        self.tp = 0  # True Positive
        self.tn = 0  # True Negative
        self.fp = 0  # False Positive
        self.fn = 0  # False Negative
        self.accuracy = 0  # Pixel Accuracy, for short PA
        self.precision = 0  # CLass Pixel Accuracy, for short CPA
        self.recall = 0
        self.iou = 0
        self.f1 = 0

    def set_confusion_matrix(self, pred: Tensor, label: Tensor) -> None:
        pred = pred2mask(pred)
        self.total += label.numel()
        self.tp += torch.sum(torch.Tensor((pred == 1) & (label == 1))).item()
        self.tn += torch.sum(torch.Tensor((pred == 0) & (label == 0))).item()
        self.fp += torch.sum(torch.Tensor((pred == 1) & (label == 0))).item()
        self.fn += torch.sum(torch.Tensor((pred == 0) & (label == 1))).item()

    def set_others(self) -> None:
        self.accuracy = (self.tp + self.tn) / self.total
        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.iou = self.tp / (self.tp + self.fp + self.fn)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    @staticmethod
    def get_precision(pred: Tensor, label: Tensor) -> float:
        tp = torch.sum(torch.Tensor((pred == 1) & (label == 1))).item()
        fp = torch.sum(torch.Tensor((pred == 1) & (label == 0))).item()
        return tp / (tp + fp) if tp + fp != 0 else 1

    def get_all(self) -> Dict[str, int | float]:
        return {
            "total": self.total,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "iou": self.iou,
            "f1": self.f1
        }
