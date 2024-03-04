from copy import deepcopy
from typing import List

from torch import nn

from models.deeplab import DeepLabV3
from test import test
from train import train
from utils.decorators import timer, repeat
from utils.loss import BinaryFocalLoss


@timer("All Tasks")
@repeat(1)
def run(
        models: List[nn.Module],
        criterions: List[nn.Module],
        mode: str = "both",
        batch_size: int = 32,
        epochs: int = 9999,
        patience: int = 15,
        learning_rate: float = 1e-3,
        load_weights: bool = False
):
    """

    Args:
        models: Put Models in a List
        criterions:
        mode: "train", "test" or "both"
        batch_size:
        epochs:
        patience:
        learning_rate:
        load_weights:

    Returns:

    """
    assert mode in ("train", "test", "both"), "mode should only be \"train\", \"test\" or \"both\"!"
    for model in models:
        model = deepcopy(model)
        for criterion in criterions:
            copy_model = deepcopy(model)
            if mode in ("train", "both"):
                train(
                    copy_model,
                    criterion,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience,
                    learning_rate=learning_rate,
                    load_weights=load_weights
                )
            if mode in ("test", "both"):
                test(
                    copy_model,
                    batch_size=batch_size,
                    out_size=(512, 512),
                    colors={
                        0: [0, 0, 0],
                        1: [59, 108, 244],
                        2: [255, 255, 255]
                    }
                )


if __name__ == "__main__":
    run(
        models=[
            DeepLabV3()
        ],

        criterions=[
            BinaryFocalLoss()
        ],

        mode="both",
        batch_size=16,
        epochs=100,
        patience=10,
        learning_rate=1e-4,
        load_weights=False
    )
