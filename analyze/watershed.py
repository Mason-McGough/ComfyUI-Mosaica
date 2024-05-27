from typing import Tuple

import cv2 as cv
import numpy as np
import torch

from ..image.convert import convert_range_to_range, float2int, np2torch, torch2np


class Watershed:
    """
    Watershed segmentation for creating label image
    """

    @classmethod
    def INPUT_TYPES(s) -> dict:
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "label_image", "lut")
    FUNCTION = "watershed"
    CATEGORY = "Mosaica/Analyze"

    def watershed(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Scale and convert to uint8 for opencv
        np_img = torch2np(image)
        np_img = convert_range_to_range(np_img, 0, 1, 0, 255).astype(np.uint8)

        gray = cv.cvtColor(np_img, cv.COLOR_RGB2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Filter noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        # Find sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # Find sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Label all markers
        _, markers = cv.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers += 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(np_img, markers)
        markers += 1

        # Convert markers to label image
        label_image = np2torch(markers).unsqueeze(-1)
        label_image = float2int(label_image)

        # Gather average colors from each region
        markers = np2torch(markers).to(torch.int32).unsqueeze(-1).repeat(1, 1, 1, 3)
        lut = torch.zeros((markers.max().item() + 1, 3), dtype=torch.float32)
        color_img = torch.zeros_like(image)
        for i in range(0, markers.max() + 1):
            mask = markers == i
            lut[i] = image[mask].view(-1, 3).mean(axis=0)
            # breakpoint()
            # color_img[mask] = lut[i]
            color_img = torch.where(mask, lut[i], color_img)

        return (color_img, label_image, lut)
