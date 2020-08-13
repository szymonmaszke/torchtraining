import torch

from .. import _base, functional


class _MeanReduction(_base.Op):
    def __init__(self, reduction=torch.mean):
        self.reduction = reduction


class _SumReduction(_base.Op):
    def __init__(self, reduction=torch.sum):
        self.reduction = reduction


class TotalOfSquares(_SumReduction):
    def forward(self, data):
        return functional.metrics.regression.total_of_squares(data, self.reduction)


class RegressionOfSquares(_SumReduction):
    def forward(self, data):
        return functional.metrics.regression.regression_of_squares(
            *data, self.reduction
        )


class SquaresOfResiduals(_SumReduction):
    def forward(self, data):
        return functional.metrics.regression.squares_of_residuals(*data, self.reduction)


class R2(_base.Op):
    def forward(self, data):
        return functional.metrics.regression.r2(*data)


class AdjustedR2(_base.Op):
    def __init__(self, p: int):
        self.p = p

    def forward(self, data):
        return functional.metrics.regression.adjusted_r2(*data, self.p)


class AbsoluteError(_MeanReduction):
    def forward(self, data):
        return functional.metrics.regression.absolute_error(*data, self.reduction)


class SquaredError(_MeanReduction):
    def forward(self, data):
        return functional.metrics.regression.squared_error(*data, self.reduction)


class SquaredLogError(_MeanReduction):
    def forward(self, data):
        return functional.metrics.regression.squared_log_error(*data, self.reduction)


class MaxError(_base.Op):
    def forward(self, data):
        return functional.metrics.regression.max_error(*data)
