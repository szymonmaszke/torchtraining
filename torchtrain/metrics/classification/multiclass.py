import abc
import collections
import typing

import torch

from ... import _base, functional


class _Reduction(_base.Op):
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

    def __init__(self, reduction=torch.mean):
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, data):
        pass


class TopK(_base.Op):
    def __init__(self, k: int, reduction=torch.mean):
        self.k = k
        self.reduction = reduction

    def forward(self, data):
        return functional.metrics.classification.multiclass.topk(*data, self.reduction)


class Accuracy(_Reduction):
    def forward(self, data):
        return functional.metrics.classification.multiclass.accuracy(
            *data, self.reduction
        )


# Basic cases


class TruePositive(_Reduction):
    def forward(self, data):
        return functional.metrics.classification.multiclass.true_positive(
            *data, self.reduction
        )


class FalsePositive(_Reduction):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_positive(
            *data, self.reduction
        )


class TrueNegative(_Reduction):
    def forward(self, data):
        return functional.metrics.classification.multiclass.true_negative(
            *data, self.reduction
        )


class FalseNegative(_Reduction):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_negative(
            *data, self.reduction
        )


# Confusion matrix


class ConfusionMatrix(_Reduction):
    def forward(self, data):
        return functional.metrics.classification.multiclass.confusion_matrix(
            *data, self.reduction
        )


class Recall(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.recall(*data)


class Specificity(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.specificity(*data)


class Precision(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.precision(*data)


class NegativePredictiveValue(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.negative_predictive_value(
            *data
        )


class FalseNegativeRate(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_negative_rate(*data)


class FalsePositiveRate(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_positive_rate(*data)


class FalseDiscoveryRate(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_discovery_rate(*data)


class FalseOmissionRate(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_omission_rate(*data)


class CriticalSuccessIndex(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.critical_success_index(
            *data
        )


class BalancedAccuracy(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.balanced_accuracy(*data)


class F1(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.f1(*data)


class FBeta(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.fbeta(*data)


class MatthewsCorrelationCoefficient(_base.Op):
    def forward(self, data):
        return functional.metrics.classification.multiclass.matthews_correlation_coefficient(
            *data
        )
