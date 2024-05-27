from typing import Tuple

import torch
from matplotlib import colormaps, colors


class LoadLUTFromMatplotlib:
    """
    Load a Look-Up Table (LUT) from matplotlib colormaps
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "colormap": (
                    [
                        c
                        for c in colormaps.keys()
                        if isinstance(colormaps[c], colors.ListedColormap)
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "load_lut_from_matplotlib"
    CATEGORY = "Mosaica/LUT"

    def load_lut_from_matplotlib(self, colormap: str) -> Tuple[torch.Tensor]:
        """
        Load a Look-Up Table (LUT) from matplotlib colormaps

        Args:
            colormap: Name of the matplotlib colormap to load

        Returns:
            torch.Tensor: The LUT as a 2D tensor
        """
        lut = colormaps[colormap]
        return (torch.tensor(lut.colors, dtype=torch.float32),)
