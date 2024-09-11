import os
from time import strftime, time

import torch
from torch.utils.data import DataLoader

from utils.dataset import MaskDataset, masks2rgb, save_tensor2mask_with_boundary, save_tensor2mask
from utils.decorators import timer
from utils.metrics import ConfusionMatrix


def get_log_path(model_info):
    """
    Get Log Path
    Args:
        model_info:

    Returns:

    """
    model_name = model_info["model"]
    today = model_info["today"]
    now_time = model_info["time"]

    model_path = f"logs/{model_name}"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    test_path = os.path.join(model_path, "test")
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    today_path = os.path.join(test_path, today)
    if not os.path.exists(today_path):
        os.mkdir(today_path)
    now_time_path = os.path.join(today_path, now_time)
    if not os.path.exists(now_time_path):
        os.mkdir(now_time_path)

    return os.path.join(now_time_path, "test_metrics.log")


def get_weights_path_dict(model_info):
    model_name = model_info["model"]
    return {
        "best": f"weights/{model_name}/best_weights.pth",
        "final": f"weights/{model_name}/final_weights.pth"
    }


def get_preds_path(model_info):
    model_name = model_info["model"]
    model_path = f"logs/{model_name}"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    preds_path = os.path.join(model_path, "preds")
    if not os.path.exists(preds_path):
        os.mkdir(preds_path)
    masks_path = os.path.join(preds_path, "masks")
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)
    rgb_path = os.path.join(preds_path, "masks_rgb")
    if not os.path.exists(rgb_path):
        os.mkdir(rgb_path)
    return {
        "preds": preds_path,
        "masks": masks_path,
        "rgb": rgb_path
    }


def save_log(model_info, *dicts):
    if len(dicts) == 1:
        with open(model_info, "w+") as file:
            for key, value in dicts[0].items():
                file.writelines(f"{key}: {value}\n")
    elif len(dicts) == 2:
        with open(f"{model_info}1", "w+") as file:
            for key, value in dicts[0].items():
                file.writelines(f"{key}: {value}\n")
        with open(f"{model_info}2", "w+") as file:
            for key, value in dicts[1].items():
                file.writelines(f"{key}: {value}\n")


@timer("Test")
def test(
        model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        data_path=r"data",
        batch_size=32,
        out_size=(512, 512),
        colors=None,
):
    """

    Args:
        model:
        device:
        data_path:
        batch_size:
        colors: Color Maps

    Returns:

    """
    print("==================== PREPARE TIME ====================")
    # 1. Print and Log Info
    model_info = {
        "model": str(model).split("(")[0],
        "today": strftime("%Y%m%d"),
        "time": strftime("%H%M")
    }

    # 2. Load Weights
    model = model.to(device)
    weights_path_dict = get_weights_path_dict(model_info)
    if os.path.exists(weights_path_dict["best"]):
        model.load_state_dict(torch.load(weights_path_dict["best"]))
        print("Load Best Weights")
    else:
        raise RuntimeError("No Weights to Load!")

    # 3. Load Datasets
    test_dataset = MaskDataset(data_path, "test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 4. Set Confusion Matrix
    confusion_matrix = ConfusionMatrix()

    # 5. Test
    print("\n==================== TEST START ====================")
    model.eval()

    time()
    with torch.no_grad():
        for image, label, name in test_loader:
            image, label = image.to(device), label.to(device)
            pred = model(image)

            # 6. Generate Test Confusion Matrix
            confusion_matrix.set_confusion_metrics(pred, label)

            # 7. Output Pred Masks
            for i in range(len(name)):
                # save_tensor2mask_with_boundary(
                #     os.path.join(get_preds_path(model_info)["masks"], name[i]),
                #     pred[i],
                #     label[i],
                #     out_size
                # )
                save_tensor2mask(
                    os.path.join(get_preds_path(model_info)["masks"], name[i]),
                    pred[i],
                    out_size
                )

    # 6. Output RGB Masks
    masks2rgb(
        get_preds_path(model_info)["preds"],
        colors
    )
    print("==================== TEST END ====================")

    # 7. Save Logs
    # BE PATIENT TO WAIT UNTIL TEST ENDS, OR LOGS WON'T BE SAVED!!!
    confusion_matrix.set_others()
    save_log(get_log_path(model_info), confusion_matrix.get_all())
