import abc
import collections
import typing

import torch

from ... import _base, functional


class _Threshold(_base.Op):
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

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    @abc.abstractmethod
    def forward(self, data):
        pass


class _ThresholdReduction(_base.Op):
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

    def __init__(self, threshold: float = 0.0, reduction=torch.mean):
        self.threshold = threshold
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, data):
        pass


# https://en.wikipedia.org/wiki/Sensitivity_and_specificity
# FPR (InverseRecall/Fall-out), Balan


class Accuracy(_ThresholdReduction):
    __doc__ = _ThresholdReduction.__doc__.format(
        header="""Calculate accuracy score between `output` and `target`.""",
    )

    def forward(self, data):
        return functional.classification.binary.accuracy(
            *data, self.threshold, self.reduction,
        )


class Jaccard(_ThresholdReduction):
    __doc__ = _ThresholdReduction.__doc__.format(
        header="""Calculate accuracy between `output` and `target`.""",
    )

    def forward(self, data):
        return functional.classification.binary.jaccard(
            *data, self.threshold, self.reduction,
        )


# Double Condition


class TruePositive(_ThresholdReduction):
    def forward(self, data):
        return functional.classification.binary.true_positive(
            *data, self.threshold, self.reduction
        )


class FalsePositive(_ThresholdReduction):
    def forward(self, data):
        return functional.classification.binary.false_positive(
            *data, self.threshold, self.reduction
        )


class TrueNegative(_ThresholdReduction):
    def forward(self, data):
        return functional.classification.binary.true_negative(
            *data, self.threshold, self.reduction
        )


class FalseNegative(_ThresholdReduction):
    def forward(self, data):
        return functional.classification.binary.false_negative(
            *data, self.threshold, self.reduction
        )


# Confusion matrix


class ConfusionMatrix(_ThresholdReduction):
    def forward(self, data):
        return functional.classification.binary.confusion_matrix(
            *data, self.threshold, self.reduction
        )


# Rate metrics


class Recall(_Threshold):
    def forward(self, data):
        return functional.classification.binary.recall(*data, self.threshold)


class Specificity(_Threshold):
    def forward(self, data):
        return functional.classification.binary.specificity(*data, self.threshold)


class Precision(_Threshold):
    def forward(self, data):
        return functional.classification.binary.precision(*data, self.threshold)


class NegativePredictiveValue(_Threshold):
    def forward(self, data):
        return functional.classification.binary.negative_predictive_value(
            *data, self.threshold
        )


class FalseNegativeRate(_Threshold):
    def forward(self, data):
        return functional.classification.binary.false_negative_rate(
            *data, self.threshold
        )


class FalsePositiveRate(_Threshold):
    def forward(self, data):
        return functional.classification.binary.false_positive_rate(
            *data, self.threshold
        )


class FalseDiscoveryRate(_Threshold):
    def forward(self, data):
        return functional.classification.binary.false_discovery_rate(
            *data, self.threshold
        )


class FalseOmissionRate(_Threshold):
    def forward(self, data):
        return functional.classification.binary.false_omission_rate(
            *data, self.threshold
        )


class F1(_Threshold):
    def forward(self, data):
        return functional.classification.binary.f1(*data, self.threshold)


class FBeta(_base.Op):
    def __init__(self, beta: float, threshold: float = 0.0):
        self.beta = beta
        self.threshold = threshold

    def forward(self, data):
        return functional.classification.binary.fbeta(*data, self.threshold)


class MatthewsCorrelationCoefficient(_Threshold):
    def forward(self, data):
        return functional.classification.binary.matthews_correlation_coefficient(
            *data, self.threshold
        )
