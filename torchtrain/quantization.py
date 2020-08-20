import torch

from ._base import Operation


class Dequantize(Operation):
    """Given a quantized Tensor, dequantize it and return an fp32 Tensor.

    Arguments
    ---------
    data: torch.Tensor
        Quantized `torch.Tensor` to dequantize.


    """

    def forward(self, data):
        return data.dequantize()
