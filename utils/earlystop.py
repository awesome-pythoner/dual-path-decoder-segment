import torch
from torch import nn


class EarlyStop:
    def __init__(self, model: nn.Module, save_path: str, patience: int = 20) -> None:
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.min_loss = float("inf")
        self.stop = False

    def __call__(self, loss: float) -> None:
        if loss > self.min_loss:
            self.counter += 1
            print(f"EarlyStop Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.min_loss = loss
            self.save_checkpoint(loss)
            self.counter = 0

    def save_checkpoint(self, loss: float) -> None:
        print(f"Best Val Loss")
        torch.save(self.model.state_dict(), self.save_path)
        self.min_loss = loss

    def is_stop(self) -> bool:
        return self.stop
