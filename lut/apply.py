from typing import Tuple

import torch

from ..image.convert import convert_range_to_range_int


class ApplyLUTToLabelImage:
    """
    Apply lookup table to label image
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label_image": ("IMAGE",),
                "lut": ("IMAGE",),
                "scale_labels_to_lut_range": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Enable",
                        "label_off": "Disable",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lut_to_label_image"
    CATEGORY = "Mosaica/LUT"

    def apply_lut_to_label_image(
        self,
        label_image: torch.Tensor,
        lut: torch.Tensor,
        scale_labels_to_lut_range: bool,
    ) -> Tuple[torch.Tensor]:
        """
        Apply lookup table to label image

        Args:
            label_image: Label image tensor
            lut: Lookup table tensor
            scale_labels_to_lut_range: If True, scale labels to LUT range
        """
        if label_image.dim() != 4:
            raise ValueError("Input image must be 3D tensor")
        if label_image.shape[3] != 1:
            raise ValueError("Input label_image must have 1 channel")
        if lut.dim() != 2:
            raise ValueError("LUT must be 2D tensor")
        if lut.shape[1] != 3:
            raise ValueError("LUT must have 3 channels")

        # Reshape label image to list of labels
        height, width = label_image.shape[1:3]
        labels = label_image.view(-1)

        if scale_labels_to_lut_range:
            labels = convert_range_to_range_int(
                labels, labels.min(), labels.max(), 0, lut.shape[0] - 1
            )

        # Apply LUT
        output_img = lut[labels, :]

        # Reshape output image
        output_img = output_img.view(1, height, width, 3)
        return (output_img,)
