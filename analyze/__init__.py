import numpy as np


def add_pixel_distance_as_channels(image: np.ndarray) -> np.ndarray:
    """
    Add pixel distance to the image

    Args:
        image: Input image with shape (H, W, C)

    Returns:
        Image with pixel distance channels with shape (H, W, C + 2)
    """
    xv, yv = np.meshgrid(
        np.arange(image.shape[1]),
        np.arange(image.shape[0]),
        indexing="xy",
    )
    xv = np.expand_dims((xv / image.shape[1] * 2) - 1.0, axis=-1)
    yv = np.expand_dims((yv / image.shape[0] * 2) - 1.0, axis=-1)
    return np.concatenate([image, xv, yv], axis=-1)
