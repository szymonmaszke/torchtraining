import abc
import collections
import typing

import torch

from . import _base

# Useful interfaces

# Precision, Recall, ConfusionMatrix, AbsoluteError, SquaredError, PairwiseDistance,
# TopK, F2, FBeta, MCC, TruePositive, FalsePositive, TrueNegative, FalseNegative,
# AreaUnderCurve, Dice


# Shared interfaces

# Multiclass & Binary?


class _Base(_base.Op):
    """{header}

    {body}

    Parameters
    ----------
    reduction : Callable, optional
        One argument callable getting tensor and outputing some value.
        Default: `torch.sum` (use `torchtrain.Mean` for correct results after saving).
    {additional_arguments}

    """

    def __init__(self, reduction=torch.mean):
        self.reduction = reduction

    def _output_target_same_shape(self, output, target):
        if output.shape != target.shape:
            raise ValueError(
                "Output and target has to be of the same shape! Got {} for output and {} for target".format(
                    output.shape, target.shape
                )
            )

    @abc.abstractmethod
    def forward(self, data):
        pass


class Accuracy(_Multiclass):
    __doc__ = _Multiclass.__doc__.format(
        header="""Calculate accuracy between `output` and `target`.""",
        body="""Works for both logits and probabilities of `output`.""",
        additional_arguments="",
    )

    def forward(self, data):
        output, target = data
        target = target.squeeze()
        if len(target.shape) != 1:
            raise ValueError()

        output = torch.argmax(output, dim=-1)
        self._output_target_same_shape(output, target)

        return torch.reduction((output == target).float())
