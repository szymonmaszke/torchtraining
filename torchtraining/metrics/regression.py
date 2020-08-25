import torch

from .. import _base, functional
from . import utils

###############################################################################
#
#                           COMMON BASE CLASSES
#
###############################################################################


class _MeanReduction(_base.Operation):
    def __init__(self, reduction=torch.mean):
        super().__init__()

        self.reduction = reduction


class _SumReduction(_base.Operation):
    def __init__(self, reduction=torch.sum):
        super().__init__()

        self.reduction = reduction


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


@utils.regression.docstring(
    header="""Absolute error between `outputs` and `targets`.""", reduction="mean",
)
class AbsoluteError(_MeanReduction):
    def forward(self, data):
        return functional.metrics.regression.absolute_error(*data, self.reduction)


@utils.regression.docstring(
    header="""Squared error between `outputs` and `targets`.""", reduction="mean",
)
class SquaredError(_MeanReduction):
    def forward(self, data):
        return functional.metrics.regression.squared_error(*data, self.reduction)


@utils.regression.docstring(
    header="""Squared log error between `outputs` and `targets`.""", reduction="mean",
)
class SquaredLogError(_MeanReduction):
    def forward(self, data):
        return functional.metrics.regression.squared_log_error(*data, self.reduction)


###############################################################################
#
#                           SUM DEFAULT REDUCTION
#
###############################################################################


@utils.regression.docstring(
    header="""Regression of squares between `outputs` and `targets`.""",
    reduction="sum",
)
class RegressionOfSquares(_SumReduction):
    def forward(self, data):
        return functional.metrics.regression.regression_of_squares(
            *data, self.reduction
        )


@utils.regression.docstring(
    header="""Square of residuals between `outputs` and `targets`.""", reduction="sum",
)
class SquaresOfResiduals(_SumReduction):
    def forward(self, data):
        return functional.metrics.regression.squares_of_residuals(*data, self.reduction)


###############################################################################
#
#                               NO REDUCTION
#
###############################################################################


@utils.regression.docstring(header="""R2 score between `outputs` and `targets`.""",)
class R2(_base.Operation):
    def forward(self, data):
        return functional.metrics.regression.r2(*data)


@utils.regression.docstring(
    header="""Maximum error between `outputs` and `targets`.""",
)
class MaxError(_base.Operation):
    def forward(self, data):
        return functional.metrics.regression.max_error(*data)


###############################################################################
#
#                               OTHER METRICS
#
###############################################################################


class TotalOfSquares(_SumReduction):
    """Total of squares of single `tensor`.

    Parameters
    ----------
    reduction: Callable, optional
        One argument callable getting `torch.Tensor` and returning `torch.Tensor`.
        Default: `torch.sum` (sum of all elements, user can use `torchtraining.savers.Sum`
        to get sum across iterations/epochs).

    """

    def forward(self, data):
        """
        Arguments
        ---------
        data: torch.Tensor
            Tensor containing `float` data of any shape. Usually `targets`.

        Returns
        -------
        torch.Tensor
            If `reduction` is left as default {} is taken and single value returned.
            Otherwise whatever `reduction` returns.
        """
        return functional.metrics.regression.total_of_squares(data, self.reduction)


class AdjustedR2(_base.Operation):
    """Adjusted R2 score between `outputs` and `targets`.

    Parameters
    ----------
    p: int
        Number of explanatory terms in model.

    """

    def __init__(self, p: int):
        super().__init__()

        self.p = p

    def forward(self, data):
        """
        Arguments
        ---------
        data: Tuple[torch.Tensor, torch.Tensor]
            Tuple containing `outputs` from neural network and regression `targets`.
            `outputs` should be of shape :math:`(N, *)`, where :math:`N` is the number of samples.
            Should contain `floating` point values.
            `targets` should be in the same shape `outputs` and be of `float` data type as well.

        Returns
        -------
        torch.Tensor
            Scalar `tensor`

        """
        return functional.metrics.regression.adjusted_r2(*data, self.p)
