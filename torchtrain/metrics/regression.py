from .. import _base
from . import functional


class _ReductionOnly(_base.Op):
    def __init__(self, reduction=torch.mean):
        self.reduction = reduction


class R2Score(_base.Op):
    pass


class AbsoluteError(_ReductionOnly):
    def forward(self, data):
        return functional.regression.absolue_error(*data, self.reduction)


class SquaredError(_ReductionOnly):
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
