import os
from typing import Dict, List
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms


def change_grey_size(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """ Transfer Image to the Same Size """
    return cv2.resize(image, size)


def change_rgb_size(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """ Transfer Image to the Same Size """
    return cv2.resize(image, size)


def mask2tensor(mask: np.ndarray) -> Tensor:
    mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(2)
    return mask.permute(2, 0, 1).to(torch.float)


def tensor2mask(tensor: Tensor) -> np.ndarray:
    return tensor.squeeze().cpu().numpy()


def save_mask(path: str, mask: np.ndarray, size: Tuple[int, int] = (256, 256)) -> None:
    cv2.imwrite(path, change_grey_size(mask, size))


def save_tensor2mask(path: str, tensor: Tensor, size: Tuple[int, int] = (256, 256)) -> None:
    mask = tensor2mask(tensor)
    cv2.imwrite(path, change_grey_size(mask, size))


def save_tensor2mask_with_boundary(path: str, pred: Tensor, label: Tensor, size: Tuple[int, int] = (256, 256)) -> None:
    mask = pred2mask_with_boundary(pred, label)
    mask = tensor2mask(mask)
    cv2.imwrite(path, change_grey_size(mask, size))


def save_tensor2grey(path: str, tensor: Tensor, size: Tuple[int, int] = (256, 256)) -> None:
    tensor = 255 * tensor
    tensor = tensor.squeeze().int().cpu().numpy()
    cv2.imwrite(path, change_grey_size(tensor, size))


def masks2rgb(path: str, colors: Dict[int, List[int]]) -> None:
    mask_dir = os.path.join(path, "masks")
    rgb_dir = os.path.join(path, "images")
    for name in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for key, value in colors.items():
            rgb[mask == key] = value
        cv2.imwrite(os.path.join(rgb_dir, name), rgb)


def pred2mask(pred: Tensor, threshold: float = 0.5) -> Tensor:
    return torch.Tensor(pred >= threshold).to(torch.float32)


def pred2mask_with_boundary(pred: Tensor, label: Tensor, threshold: float = 0.5) -> Tensor:
    boundary: Tensor = LaplacianConv().to(label.device)(label)
    mask = torch.Tensor(pred >= threshold).to(torch.float32)
    mask[boundary == 1] = 2
    return mask


def mask2edge(mask: np.ndarray, save_path: str) -> None:
    tensor = mask2tensor(mask)
    edge = LaplacianConv()(tensor)
    edge = 255 * tensor2mask(edge)
    cv2.imwrite(save_path, edge)


class LaplacianConv(nn.Module):
    def __init__(self, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        self.kernel = nn.Parameter(
            torch.tensor([[[[1., 1., 1.],
                            [1., -8., 1.],
                            [1., 1., 1.]]]], dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=1)
        return torch.Tensor(x != 0).to(torch.float32)


class MaskDataset(Dataset):
    """
    The Path of Dataset Must be Like:
    --any
    ----train
    ------images
    --------name.png
    ------masks
    --------name.png
    ----test
    ------images
    --------name.png
    ------masks
    --------name.png
    """

    def __init__(self, path: str, mode: str) -> None:
        """

        Args:
            path:
            mode: train or val or test
        """
        assert mode in ("train", "val", "test"), "Mode Must be \"train\", \"val\" or \"test\"!"
        self.data_path = os.path.join(path, mode)
        self.image_path = os.path.join(self.data_path, "images")
        self.label_path = os.path.join(self.data_path, "masks")
        self.names = os.listdir(self.label_path)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        name = self.names[index]
        label_path = os.path.join(self.label_path, name)
        image_path = os.path.join(self.image_path, name)
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # Here to Change the Resized Images Size
        image, label = change_rgb_size(image, (256, 256)), change_grey_size(label, (256, 256))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225), True)
        ])
        return transform(image), mask2tensor(label), name
