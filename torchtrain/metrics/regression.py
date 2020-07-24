import torch

from .. import _base
from . import functional


class _MeanReduction(_base.Op):
    def __init__(self, reduction=torch.mean):
        self.reduction = reduction


class _SumReduction(_base.Op):
    def __init__(self, reduction=torch.sum):
        self.reduction = reduction


class TotalOfSquares(_SumReduction):
    def forward(self, data):
        return functional.regression.regression_of_squares(data, self.reduction)


class RegressionOfSquares(_SumReduction):
    def forward(self, data):
        return functional.regression.regression_of_squares(*data, self.reduction)


class SquaresOfResiduals(_SumReduction):
    def forward(self, data):
        return functional.regression.squares_of_residuals(*data, self.reduction)


class R2(_base.Op):
    def forward(self, data):
        return functional.regression.r2(*data)


class AdjustedR2(_base.Op):
    def __init__(self, p: int):
        self.p = p

    def forward(self, data):
        return functional.regression.adjusted_r2(*data, self.p)


class AbsoluteError(_MeanReduction):
    def forward(self, data):
        return functional.regression.absolute_error(*data, self.reduction)


class SquaredError(_MeanReduction):
    def forward(self, data):
        return functional.regression.squared_error(*data, self.reduction)


class PairwiseDistance(_base.Op):
    def __init__(self, p: float = 2.0, eps: float = 1e-06, reduction=torch.mean):
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, data):
        return functional.regression.pairwise_distance(
            *data, self.p, self.eps, self.reduction,
        )
