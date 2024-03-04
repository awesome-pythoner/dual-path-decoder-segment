import numpy as np


def warmup_cosine_lr(warmup_epochs: int = 10, total_epochs: int = 50):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return lr_lambda
