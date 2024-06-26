from typing import Literal, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import MeanShift as SklearnMeanShift

from ..analyze import add_pixel_distance_as_channels
from ..image.convert import (
    convert_range_to_range,
    convert_to_normalized_colorspace,
    np2torch,
    torch2np,
)


class MeanShift:
    """
    MeanShift clustering algorithm for image segmentation
    """

    @classmethod
    def INPUT_TYPES(s) -> dict:
        """
        Return a dictionary which contains config for all input fields.

        See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "bandwidth": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "color_space": (["RGB", "LAB"],),
                "use_pixel_distance": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Enable",
                        "label_off": "Disable",
                    },
                ),
                "max_iter": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "label_image", "lut")
    FUNCTION = "meanshift"
    CATEGORY = "Mosaica/Analyze"

    def meanshift(
        self,
        image: torch.Tensor,
        bandwidth: float,
        color_space: Literal["RGB", "LAB"],
        use_pixel_distance: str,
        max_iter: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform MeanShift clustering on an image

        Args:
            image: Input image tensor from ComfyUI node
            bandwidth: Bandwidth for the MeanShift algorithm. Higher values are faster but result
                in fewer colors.
            color_space: Color space to use for clustering. Either "RGB" or "LAB".
            use_pixel_distance: Whether to include pixel distance in the clustering. If enabled,
                the image will be extended with pixel coordinates.
            max_iter: Maximum number of iterations for the MeanShift algorithm.

        Returns:
            Output image, label image, and lookup table
        """
        # convert and load image to numpy
        np_img = torch2np(image)
        norm_img = convert_to_normalized_colorspace(np_img, color_space)

        # add pixel distance to the image
        n_output_channels = 3
        if use_pixel_distance == "enable":
            norm_img = add_pixel_distance_as_channels(norm_img)
            n_output_channels = 5

        # apply clustering to gather labels
        clustering = SklearnMeanShift(
            bandwidth=bandwidth, bin_seeding=True, max_iter=max_iter
        )
        labels = clustering.fit_predict(norm_img.reshape(-1, n_output_channels))

        # map labels to colors
        mapped_img = clustering.cluster_centers_[labels, :3]
        mapped_img = mapped_img.reshape(np_img.shape)

        # convert back to RGB if needed
        if color_space == "LAB":
            mapped_img = convert_range_to_range(mapped_img, -1, 1, 0.0, 255.0)
            mapped_img = cv2.cvtColor(mapped_img.astype(np.float32), cv2.COLOR_LAB2RGB)

        # convert to torch and return
        output_img = np2torch(mapped_img)
        label_img = np2torch(labels.reshape((*np_img.shape[:2], 1))).int()
        lut = torch.from_numpy(clustering.cluster_centers_[:, :3])
        return (output_img, label_img, lut)
