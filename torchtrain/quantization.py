import torch

from ._base import Op


class Dequantize(Op):
    def forward(self, data):
        return data.dequantize()
