"""Quantization related operations."""

import torch

from ._base import Operation


class Dequantize(Operation):
    """Given a quantized Tensor, dequantize it and return `float` Tensor.

    Arguments
    ---------
    data: torch.Tensor
        Quantized `torch.Tensor` to dequantize.



    """

    def forward(self, data):
        """
        Returns
        ---------
        data: torch.Tensor
            Dequantized to `float32` `torch.Tensor`
        """

        return data.dequantize()
