import torch

from .. import _base, functional


class Cosine(_base.Op):
    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon

    def forward(self, data):
        return functional.distance.cosine(*data, self.epsilon)


class Euclidean(_base.Op):
    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon

    def forward(self, data):
        return functional.distance.euclidean(*data, self.epsilon)
