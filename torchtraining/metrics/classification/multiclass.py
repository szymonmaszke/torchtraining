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


class _ReductionSum(_base.Operation):
    def __init__(self, reduction=torch.sum):
        super().__init__()
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, data):
        pass


class _ReductionMean(_base.Operation):
    def __init__(self, reduction=torch.mean):
        super().__init__()
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


@utils.multiclass.docs(
    header="""Calculate accuracy score between `output` and `target`.""",
    reduction="mean",
)
class Accuracy(_ReductionMean):
    def forward(self, data):
        return functional.metrics.classification.multiclass.accuracy(
            *data, self.reduction
        )


@utils.multiclass.docs(
    header="""Calculate Jaccard score between `output` and `target`.""",
    reduction="mean",
)
class Jaccard(_ReductionMean):
    def forward(self, data):
        return functional.metrics.classification.multiclass.jaccard(
            *data, self.reduction
        )


###############################################################################
#
#                           SUM DEFAULT REDUCTION
#
###############################################################################


# Basic cases
@utils.multiclass.docs(
    header="""Number of false positives between `output` and `target`.""",
    reduction="sum",
)
class TruePositive(_ReductionSum):
    def forward(self, data):
        return functional.metrics.classification.multiclass.true_positive(
            *data, self.reduction
        )


@utils.multiclass.docs(
    header="""Number of false positives between `output` and `target`.""",
    reduction="sum",
)
class FalsePositive(_ReductionSum):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_positive(
            *data, self.reduction
        )


@utils.multiclass.docs(
    header="""Number of true negatives between `output` and `target`.""",
    reduction="sum",
)
class TrueNegative(_ReductionSum):
    def forward(self, data):
        return functional.metrics.classification.multiclass.true_negative(
            *data, self.reduction
        )


@utils.multiclass.docs(
    header="""Number of false negatives between `output` and `target`.""",
    reduction="sum",
)
class FalseNegative(_ReductionSum):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_negative(
            *data, self.reduction
        )


@utils.multiclass.docs(
    header="""Confusion matrix between `output` and `target`.""", reduction="sum",
)
class ConfusionMatrix(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.confusion_matrix(*data,)


###############################################################################
#
#                               NO REDUCTION
#
###############################################################################


@utils.multiclass.docs(header="""Recall between `output` and `target`.""",)
class Recall(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.recall(*data)


@utils.multiclass.docs(header="""Specificity between `output` and `target`.""",)
class Specificity(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.specificity(*data)


@utils.multiclass.docs(header="""Precision between `output` and `target`.""",)
class Precision(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.precision(*data)


@utils.multiclass.docs(
    header="""Negative predictive value between `output` and `target`.""",
)
class NegativePredictiveValue(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.negative_predictive_value(
            *data
        )


@utils.multiclass.docs(header="""False negative rate between `output` and `target`.""",)
class FalseNegativeRate(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_negative_rate(*data)


@utils.multiclass.docs(header="""False positive rate between `output` and `target`.""",)
class FalsePositiveRate(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_positive_rate(*data)


@utils.multiclass.docs(
    header="""False discovery rate between `output` and `target`.""",
)
class FalseDiscoveryRate(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_discovery_rate(*data)


@utils.multiclass.docs(header="""False omission rate between `output` and `target`.""",)
class FalseOmissionRate(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.false_omission_rate(*data)


@utils.multiclass.docs(
    header="""Critical success index between `output` and `target`.""",
)
class CriticalSuccessIndex(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.critical_success_index(
            *data
        )


@utils.multiclass.docs(
    header="""Critical success index between `output` and `target`.""",
)
class BalancedAccuracy(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.balanced_accuracy(*data)


@utils.multiclass.docs(header="""F1 score between `output` and `target`.""",)
class F1(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.f1(*data)


@utils.multiclass.docs(
    header="""Matthews correlation coefficient between `output` and `target`.""",
)
class MatthewsCorrelationCoefficient(_base.Operation):
    def forward(self, data):
        return functional.metrics.classification.multiclass.matthews_correlation_coefficient(
            *data
        )


###############################################################################
#
#                               OTHER METRICS
#
###############################################################################


class FBeta(_base.Operation):
    r"""Get f-beta score between `outputs` and `targets`.

    Works for both logits and probabilities of `output` out of the box.

    Parameters
    ----------
    beta: float
        Beta coefficient of `f-beta` score.

    Arguments
    ---------
    data: Tuple[torch.Tensor, torch.Tensor]
        Tuple containing `outputs` from neural network and `targets` (ground truths).
        `outputs` should be of shape :math:`(N, *, C-1)`, where :math:`C` is the number of classes.
        Should contain `logits` (unnormalized probabilities) or `probabilities` after
        `softmax` activation or similar.
        `targets` should be of shape :math:`(N, *)` and contain integers in the range
        :math:`[0, C-1]`


    Returns
    -------
    torch.Tensor
        Scalar `tensor`

    """

    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, data):
        return functional.metrics.classification.multiclass.f_beta(*data, self.beta)


class TopK(_base.Operation):
    r"""Get top-k accuracy score between `outputs` and `targets`.

    Works for both logits and probabilities of `output` out of the box.

    Parameters
    ----------
    k: int
        How many top results should be chosen.
    reduction: Callable, optional
        One argument callable getting `torch.Tensor` and returning `torch.Tensor`.
        Default: `torch.sum` (sum of all elements, user can use `torchtraining.savers.Sum`
        to get sum across iterations/epochs).

    Arguments
    ---------
    data: Tuple[torch.Tensor, torch.Tensor]
        Tuple containing `outputs` from neural network and `targets` (ground truths).
        `outputs` should be of shape :math:`(N, *, C-1)`, where :math:`C` is the number of classes.
        Should contain `logits` (unnormalized probabilities) or `probabilities` after
        `softmax` activation or similar.
        `targets` should be of shape :math:`(N, *)` and contain integers in the range
        :math:`[0, C-1]`

    Returns
    -------
    torch.Tensor
        If `reduction` is left as default mean is taken and single value returned.
        Otherwise whatever `reduction` returns.

    """

    def __init__(self, k: int, reduction=torch.mean):
        super().__init__()
        self.k = k
        self.reduction = reduction

    def forward(self, data):
        return functional.metrics.classification.multiclass.topk(*data, self.reduction)
