import torch

from .. import _base, functional
from . import utils

###############################################################################
#
#                               NO REDUCTION
#
###############################################################################


@utils.distance.docstring(header="Cosine distance between `outputs` and `targets`")
class Cosine(_base.Operation):
    def forward(self, data):
        return functional.metrics.distance.cosine(*data)


@utils.distance.docstring(header="Euclidean distance between `outputs` and `targets`")
class Euclidean(_base.Operation):
    def forward(self, data):
        return functional.metrics.distance.euclidean(*data)


###############################################################################
#
#                              OTHER METRICS
#
###############################################################################


class Pairwise(_base.Operation):
    """Computes the batchwise pairwise distance between vectors :math:`v_1`, :math:`v_2` using specified norm.

    .. math ::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.


    Parameters
    ----------
    p: float, optional
        Degree of `norm`. Default: `2`
    eps: float, optional
        Epsilon to avoid division by zero. Default: `1e-06`
    reduction: Callable(torch.Tensor) -> Any, optional
        One argument callable getting `torch.Tensor` and returning argument
        after specified reduction.
        Default: `torch.mean` (mean across batch, user can use `torchtrain.savers.Mean`
        to get mean across iterations/epochs).

    Arguments
    ---------
    data: Tuple[torch.Tensor, torch.Tensor]
        Tuple containing `outputs` from neural network and regression `targets`.
        `outputs` should be of shape :math:`(N, F)`, where :math:`N` is the number of samples,
        :math:`F` is the number of features.
        Should contain `floating` point values.
        `targets` should be in the same shape `outputs` and be of `float` data type as well.

    Returns
    -------
    torch.Tensor
        If `reduction` is left as default {} is taken and single value returned.
        Otherwise whatever `reduction` returns.

    """

    def __init__(self, p: float = 2.0, eps: float = 1e-06, reduction=torch.mean):
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, data):
        return functional.metrics.distance.pairwise(
            *data, self.p, self.eps, self.reduction,
        )
