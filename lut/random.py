import torch


class RandomLUT:
    """
    Generate a random Look-Up Table (LUT) of RGB colors in range [0, 1]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_rows": (
                    "INT",
                    {
                        "default": 256,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "random_lut"
    CATEGORY = "Mosaica/LUT"

    def random_lut(self, num_rows: int) -> torch.Tensor:
        """
        Generate a random Look-Up Table (LUT) of RGB colors

        Args:
            num_rows: Number of rows in the LUT

        Returns:
            The LUT as a 2D tensor of shape [num_rows, 3] in range [0, 1]
        """
        return (torch.rand((num_rows, 3), dtype=torch.float32),)
