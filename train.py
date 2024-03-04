import os
import shutil
from time import time, strftime

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from utils.dataset import MaskDataset
from utils.decorators import timer
from utils.earlystop import EarlyStop
from utils.io import Printer
from utils.loss import BinaryFocalLoss
from utils.metrics import ConfusionMatrix
from utils.schedule import warmup_cosine_lr


def get_path_dict(model_info, mode):
    """
    Get Dict of Weights Path or Log Path
    Args:
        model_info:
        mode: "weights" or "logs"

    Returns:

    """
    assert mode in ("weights", "logs"), "mode should only be \"weights\" or \"logs\"!"
    model_name = model_info["model"]
    today = model_info["today"]
    now_time = model_info["time"]

    model_path = f"{mode}/{model_name}"
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    today_path = None
    if mode == "weights":
        today_path = os.path.join(model_path, today)
    elif mode == "logs":
        train_path = os.path.join(model_path, "train")
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        today_path = os.path.join(train_path, today)
    if today_path is not None and not os.path.exists(today_path):
        os.mkdir(today_path)
    now_time_path = os.path.join(today_path, now_time)
    if not os.path.exists(now_time_path):
        os.mkdir(now_time_path)

    if mode == "weights":
        return {
            "best": f"{model_path}/best_weights.pth",
            "best backup": f"{now_time_path}/best_weights.pth",
        }
    elif mode == "logs":
        return {
            "losses": f"{now_time_path}/losses.log",
            "train precisions": f"{now_time_path}/train_precisions.log",
            "val losses": f"{now_time_path}/val_losses.log",
            "val precisions": f"{now_time_path}/val_precisions.log"
        }


def backup_weights(weights_path_dict):
    """ Backup Old Weights """
    if os.path.exists(weights_path_dict["best"]):
        shutil.copy(weights_path_dict["best"], weights_path_dict["best backup"])


def save_logs(logs_path_dict, metrics, mode):
    """
    Save Train Metrics into Logs
    Args:
        logs_path_dict:
        metrics:
        mode: TODO: "iter losses", "epoch losses", "train acc" or "val acc"

    Returns:

    """
    assert mode in logs_path_dict.keys(), \
        "mode should only be \"iter losses\", \"epoch losses\", \"train acc\" or \"val acc\"!"  # TODO
    with open(logs_path_dict[mode], "w+") as file:
        for metric in metrics:
            file.writelines(f"{metric}\n")


def batch_save_logs(log_path_dict, losses_dict):
    for key, value in losses_dict.items():
        save_logs(log_path_dict, value, key)


@timer("Train")
def train(
        model: nn.Module,
        criterion: nn.Module = BinaryFocalLoss(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        data_path: str = r"data",
        batch_size: int = 16,
        epochs: int = 9999,
        patience: int = 15,
        learning_rate: float = 1e-3,
        load_weights: bool = False
) -> None:
    print("==================== PREPARE TIME ====================")
    # 1. Print and Log Info
    model_info = {
        "model": str(model).split("(")[0],
        "today": strftime("%Y%m%d"),
        "time": strftime("%H%M")
    }

    # 2. Load Weights
    model = model.to(device)
    weights_path_dict = get_path_dict(model_info, "weights")
    if load_weights:
        if os.path.exists(weights_path_dict["best"]):
            model.load_state_dict(torch.load(weights_path_dict["best"]))
            print("Load Best Weights")
        else:
            print("No Weights to Load!")

    # 3. Load Datasets
    train_dataset = MaskDataset(data_path, "train")
    val_dataset = MaskDataset(data_path, "val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_len, val_len = len(train_loader), len(val_loader)

    # 4. Set Loss Function
    criterion = criterion

    # 5. Set Optimizer
    optimizer = optim.Adam([
        {"params": model.parameters()}
    ], lr=learning_rate, weight_decay=1e-4)
    schedule = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr(10, 50))
    early_stop = EarlyStop(model, weights_path_dict["best"], patience)

    # 6. Set Confusion Matrix

    # 7. Train
    print("\n==================== TRAIN START ====================")
    losses, val_losses = [], []
    train_precisions, val_precisions = [], []
    train_confusion_matrix, val_confusion_matrix = ConfusionMatrix(), ConfusionMatrix()

    for epoch in range(epochs):
        epoch_start_time = time()
        total_loss = 0
        total_precision = 0
        model.train()
        for image, label, _ in train_loader:
            # 8. Forward
            image, label = image.to(device), label.to(device)
            pred = model(image)

            # 9. Generate Train Confusion Matrix
            train_confusion_matrix.set_confusion_matrix(pred, label)
            total_precision += train_confusion_matrix.get_precision(pred, label)

            # 10. Calculate Loss
            loss = criterion(pred, label)
            total_loss += loss.item()

            # 11. Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

        losses.append(total_loss / train_len)
        train_precisions.append(total_precision)

        epoch_end_time = time()
        Printer.print_using_time(f"Epoch {epoch + 1}", epoch_start_time, epoch_end_time)
        print(f"Train Loss: {losses[-1]}")

        # 12. Val
        val_start_time = time()
        total_loss = 0
        total_precision = 0
        model.eval()
        with torch.no_grad():
            for image, label, _ in val_loader:
                # 13. Forward
                image, label = image.to(device), label.to(device)
                pred = model(image)

                # 14. Generate Val Confusion Matrix
                val_confusion_matrix.set_confusion_matrix(pred, label)
                total_precision += val_confusion_matrix.get_precision(pred, label)

                # 15. Get Loss
                loss = criterion(pred, label)
                total_loss += loss.item()

        val_losses.append(total_loss / val_len)
        val_precisions.append(total_precision)

        val_end_time = time()
        Printer.print_using_time("Val", val_start_time, val_end_time)
        print(f"Val Loss: {val_losses[-1]}")

        # 16. Early Stop
        early_stop(total_loss)
        print()
        if early_stop.is_stop():
            print("Early Stop!")
            break
    print("==================== TRAIN END ====================")

    # 17. Save Weights and Logs
    # BE PATIENT TO WAIT UNTIL TRAIN ENDS, OR LOGS WON'T BE SAVED!!!
    backup_weights(weights_path_dict)
    batch_save_logs(
        get_path_dict(model_info, "logs"),
        {
            "losses": losses,
            "train precisions": train_precisions,
            "val losses": val_losses,
            "val precisions": val_precisions
        }
    )
