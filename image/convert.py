import numpy as np
import torch


def convert_range_to_range(
    value: torch.Tensor | np.ndarray,
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
) -> torch.Tensor | np.ndarray:
    """
    Convert value from range [old_min, old_max] to range [new_min, new_max]

    Args:
        value: Tensor to apply range conversion to
        old_min: Old minimum value
        old_max: Old maximum value
        new_min: New minimum value
        new_max: New maximum value

    Returns:
        Converted tensor
    """
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def convert_range_to_range_int(
    value: torch.Tensor | np.ndarray,
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
) -> torch.Tensor | np.ndarray:
    """
    Scale integers to new range
    """
    adjusted = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return adjusted.round().to(torch.int32)


def np2torch(np_img: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image to torch image

    Args:
        np_img: Numpy image with shape (H, W, 3)

    Returns:
        Torch image with shape (1, H, W, 3)
    """
    t = torch.from_numpy(np_img).unsqueeze(0)
    return t.float()


def torch2np(t_img: torch.Tensor) -> np.ndarray:
    """
    Convert torch image to numpy image

    Args:
        t_img: Torch image with shape (1, H, W, 3)

    Returns:
        Numpy image with shape (H, W, 3)
    """
    np_img = t_img.squeeze(0).cpu().numpy()
    return np_img
