from typing import Literal

import cv2 as cv
import numpy as np
import torch


def float2int(t: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Convert float tensor to int tensor

    Args:
        t: Float tensor to convert to int

    Returns:
        Converted tensor
    """
    t = t.round()
    return t.to(torch.int32) if isinstance(t, torch.Tensor) else t.astype(np.int32)


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
    return float2int(adjusted)


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


def convert_to_normalized_colorspace(
    image: np.ndarray,
    color_space: Literal["RGB", "LAB"] = "RGB",
) -> np.ndarray:
    """
    Convert image to normalized color space

    Args:
        image: Input floating point image in range [0, 1]
        color_space: Color space to convert to. Either "RGB" or "LAB"

    Returns:
        Normalized image in range [-1, 1]

    Raises:
        ValueError: If invalid color space is provided
    """
    if color_space == "LAB":
        norm_img = cv.cvtColor(image, cv.COLOR_RGB2LAB)
        norm_img = convert_range_to_range(norm_img.astype(float), 0, 255, -1, 1)
    elif color_space == "RGB":
        norm_img = convert_range_to_range(image, 0, 1.0, -1.0, 1.0)
    else:
        raise ValueError(f"Invalid color space: {color_space}")

    return norm_img
