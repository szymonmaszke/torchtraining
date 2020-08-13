import torch

from .. import _base, functional


class Cosine(_base.Op):
    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon

    def forward(self, data):
        return functional.metrics.distance.cosine(*data, self.epsilon)


class Euclidean(_base.Op):
    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon

    def forward(self, data):
        return functional.metrics.distance.euclidean(*data, self.epsilon)


class Pairwise(_base.Op):
    def __init__(self, p: float = 2.0, eps: float = 1e-06, reduction=torch.mean):
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, data):
        return functional.metrics.distance.pairwise(
            *data, self.p, self.eps, self.reduction,
        )
