import abc
import collections
import typing

import torch

from .. import _base
from . import functional

# Useful interfaces

# Precision, Recall, ConfusionMatrix, AbsoluteError, SquaredError, PairwiseDistance,
# TopK, F2, FBeta, MCC, TruePositive, FalsePositive, TrueNegative, FalseNegative,
# AreaUnderCurve, Dice


class _Base(_base.Op):
    """{header}

    Works for both logits and probabilities of `output`.

    If `output` is tensor after activation (e.g. `sigmoid` or `softmax` as last neural network layer),
    user should change `threshold` to `0.5` for
    correct results (default `0.0` corresponds to unnormalized probability a.k.a logits).

    Parameters
    ----------
    reduction : Callable, optional
        One argument callable getting tensor and outputing some value.
        Default: `torch.sum` (use `torchtrain.Mean` for correct results after saving).
    threshold : float, optional
        Threshold above which prediction is considered to be positive.
        Default: `0.0`

    """

    def __init__(self, reduction=torch.mean, threshold: float = 0.0):
        self.reduction = reduction
        self.threshold = threshold

    @abc.abstractmethod
    def forward(self, data):
        pass


class Accuracy(_Base):
    __doc__ = _Base.__doc__.format(
        header="""Calculate accuracy score between `output` and `target`.""",
    )

    def forward(self, data):
        return functional.binary.accuracy(*data, self.reduction, self.threshold)


class Jaccard(_Base):
    __doc__ = _Base.__doc__.format(
        header="""Calculate accuracy between `output` and `target`.""",
    )

    def forward(self, data):
        return functional.binary.jaccard(*data, self.reduction, self.threshold)


class TruePositive(_Base):
    def forward(self, data):
        return functional.binary.true_positive(*data, self.reduction, self.threshold)


class FalsePositive(_base.Op):
    def forward(self, data):
        return functional.binary.false_positive(*data, self.reduction, self.threshold)


class TrueNegative(_base.Op):
    def forward(self, data):
        return functional.binary.true_negative(*data, self.reduction, self.threshold)


class FalseNegative(_base.Op):
    def forward(self, data):
        return functional.binary.false_negative(*data, self.reduction, self.threshold)


class ConfusionMatrix(_base.Op):
    def forward(self, data):
        return functional.binary.confusion_matrix(*data, self.reduction, self.threshold)


class Recall(_base.Op):
    def forward(self, data):
        return functional.binary.recall(*data, self.reduction, self.threshold)


class Specificity(_base.Op):
    def forward(self, data):
        return functional.binary.specificity(*data, self.reduction, self.threshold)


class Precision(_base.Op):
    def forward(self, data):
        return functional.binary.precision(*data, self.reduction, self.threshold)


class AreaUnderCurve(_base.Op):
    pass


class TopK(_base.Op):
    pass


class F2(_base.Op):
    pass


class FBeta(_base.Op):
    pass


class MatthewsCorrelationCoefficient(_base.Op):
    pass


class Dice(_base.Op):
    pass
