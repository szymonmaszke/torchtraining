import abc
import collections
import typing

import torch

from ... import _base, functional
from . import utils

###############################################################################
#
#                           COMMON BASE CLASSES
#
###############################################################################


class _Threshold(_base.Operation):
    def __init__(self, threshold: float = 0.0):
        super().__init__()

        self.threshold = threshold

    @abc.abstractmethod
    def forward(self, data):
        pass


class _ThresholdReductionMean(_base.Operation):
    def __init__(self, threshold: float = 0.0, reduction=torch.mean):
        super().__init__()

        self.threshold = threshold
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, data):
        pass


class _ThresholdReductionSum(_base.Operation):
    def __init__(self, threshold: float = 0.0, reduction=torch.sum):
        super().__init__()

        self.threshold = threshold
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, data):
        pass


###############################################################################
#
#                          CONCRETE IMPLEMENTATIONS
#
###############################################################################


###############################################################################
#
#                           MEAN DEFAULT REDUCTION
#
###############################################################################


@utils.binary.docs(
    header="""Calculate accuracy score between `output` and `target`.""",
    reduction="mean",
)
class Accuracy(_ThresholdReductionMean):
    def forward(self, data):
        return functional.metrics.classification.binary.accuracy(
            *data, self.threshold, self.reduction,
        )


@utils.binary.docs(
    header="""Calculate jaccard score between `output` and `target`.""",
    reduction="mean",
)
class Jaccard(_ThresholdReductionMean):
    def forward(self, data):
        return functional.metrics.classification.binary.jaccard(
            *data, self.threshold, self.reduction,
        )


###############################################################################
#
#                           SUM DEFAULT REDUCTION
#
###############################################################################


@utils.binary.docs(
    header="""Number of true positives between `output` and `target`.""",
    reduction="sum",
)
class TruePositive(_ThresholdReductionSum):
    def forward(self, data):
        return functional.metrics.classification.binary.true_positive(
            *data, self.threshold, self.reduction
        )


@utils.binary.docs(
    header="""Number of false positives between `output` and `target`.""",
    reduction="sum",
)
class FalsePositive(_ThresholdReductionSum):
    def forward(self, data):
        return functional.metrics.classification.binary.false_positive(
            *data, self.threshold, self.reduction
        )


@utils.binary.docs(
    header="""Number of true negatives between `output` and `target`.""",
    reduction="sum",
)
class TrueNegative(_ThresholdReductionSum):
    def forward(self, data):
        return functional.metrics.classification.binary.true_negative(
            *data, self.threshold, self.reduction
        )


@utils.binary.docs(
    header="""Number of false negatives between `output` and `target`.""",
    reduction="sum",
)
class FalseNegative(_ThresholdReductionSum):
    def forward(self, data):
        return functional.metrics.classification.binary.false_negative(
            *data, self.threshold, self.reduction
        )


@utils.binary.docs(
    header="""Confusion matrix between `output` and `target`.""", reduction="sum",
)
class ConfusionMatrix(_ThresholdReductionSum):
    def forward(self, data):
        return functional.metrics.classification.binary.confusion_matrix(
            *data, self.threshold, self.reduction
        )


###############################################################################
#
#                               NO REDUCTION
#
###############################################################################


@utils.binary.docs(header="""Recall between `output` and `target`.""",)
class Recall(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.recall(*data, self.threshold)


@utils.binary.docs(header="""Specificity between `output` and `target`.""",)
class Specificity(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.specificity(
            *data, self.threshold
        )


@utils.binary.docs(header="""Precision between `output` and `target`.""",)
class Precision(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.precision(*data, self.threshold)


@utils.binary.docs(
    header="""Negative predictive value between `output` and `target`.""",
)
class NegativePredictiveValue(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.negative_predictive_value(
            *data, self.threshold
        )


@utils.binary.docs(header="""False negative rate between `output` and `target`.""",)
class FalseNegativeRate(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.false_negative_rate(
            *data, self.threshold
        )


@utils.binary.docs(header="""False positive rate between `output` and `target`.""",)
class FalsePositiveRate(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.false_positive_rate(
            *data, self.threshold
        )


@utils.binary.docs(header="""False discovery rate between `output` and `target`.""",)
class FalseDiscoveryRate(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.false_discovery_rate(
            *data, self.threshold
        )


@utils.binary.docs(header="""False omission rate between `output` and `target`.""",)
class FalseOmissionRate(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.false_omission_rate(
            *data, self.threshold
        )


@utils.binary.docs(header="""Critical success index between `output` and `target`.""",)
class CriticalSuccessIndex(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.critical_success_index(
            *data, self.threshold
        )


@utils.binary.docs(header="""Critical success index between `output` and `target`.""",)
class BalancedAccuracy(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.balanced_accuracy(
            *data, self.threshold
        )


@utils.binary.docs(header="""F1 score between `output` and `target`.""",)
class F1(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.f1(*data, self.threshold)


@utils.binary.docs(
    header="""Matthews correlation coefficient between `output` and `target`.""",
)
class MatthewsCorrelationCoefficient(_Threshold):
    def forward(self, data):
        return functional.metrics.classification.binary.matthews_correlation_coefficient(
            *data, self.threshold
        )


###############################################################################
#
#                               OTHER METRICS
#
###############################################################################


class FBeta(_base.Operation):
    r"""Get f-beta score between `outputs` and `targets`.


    Works for both logits and probabilities of `output`.

    If `output` is tensor after `sigmoid` activation
    user should change `threshold` to `0.5` for
    correct results (default `0.0` corresponds to unnormalized probability a.k.a logits).

    Parameters
    ----------
    beta: float
        Beta coefficient of `f-beta` score.
    threshold : float, optional
        Threshold above which prediction is considered to be positive.
        Default: `0.0`

    Arguments
    ---------
    data: Tuple[torch.Tensor, torch.Tensor]
        Tuple containing `outputs` from neural network and `targets` (ground truths).
        `outputs` should be of shape :math:`(N, *)` and contain `logits` or `probabilities`.
        `targets` should be of shape :math:`(N, *)` as well and contain `boolean` values
        (or integers from set :math:`{0, 1}`).

    Returns
    -------
    torch.Tensor
        Scalar `tensor`

    """

    def __init__(self, beta: float, threshold: float = 0.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, data):
        return functional.metrics.classification.binary.f_beta(
            *data, self.beta, self.threshold
        )
