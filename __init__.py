from .analyze.meanshift import MeanShift
from .analyze.watershed import Watershed
from .lut.apply import ApplyLUTToLabelImage
from .lut.matplotlib import LoadLUTFromMatplotlib
from .lut.random import RandomLUT

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MeanShift": MeanShift,
    "Watershed": Watershed,
    "ApplyLUTToLabelImage": ApplyLUTToLabelImage,
    "RandomLUT": RandomLUT,
    "LoadLUTFromMatplotlib": LoadLUTFromMatplotlib,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeanShift": "Mean Shift",
    "Watershed": "Watershed",
    "ApplyLUTToLabelImage": "Apply LUT To Label Image",
    "RandomLUT": "Random LUT",
    "LoadLUTFromMatplotlib": "Load LUT From Matplotlib",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
