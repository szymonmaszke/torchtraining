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
    """Returns cosine distance between :math:`x_1` and :math:`x_2`, computed along dim.

    It is equal to :math:`1 - S`, where :math:`S` is cosine similarity.

    Parameters
    ----------
    dim: int, optional
        Dimension where cosine similarity is computed. Default: 1
        Degree of `norm`. Default: `2`
    eps: float, optional
        Epsilon to avoid division by zero. Default: `1e-08`
    reduction: Callable(torch.Tensor) -> Any, optional
        One argument callable getting `torch.Tensor` and returning argument
        after specified reduction.
        Default: `torch.mean` (mean across batch, user can use `torchtraining.savers.Mean`
        to get mean across iterations/epochs).


    """

    def __init__(
        self, dim: int = 1, eps: float = 1e-08, reduction=torch.mean,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, data):
        """
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
        return functional.metrics.distance.cosine(
            *data, self.dim, self.eps, self.reduction
        )


###############################################################################
#
#                              OTHER METRICS
#
###############################################################################


class Pairwise(_base.Operation):
    """Computes the batchwise pairwise distance between vectors :math:`v_1`, :math:`v_2` using specified norm.

    Parameters
    ----------
    p: float, optional
        Degree of `norm`. Default: `2`
    eps: float, optional
        Epsilon to avoid division by zero. Default: `1e-06`
    reduction: Callable(torch.Tensor) -> Any, optional
        One argument callable getting `torch.Tensor` and returning argument
        after specified reduction.
        Default: `torch.mean` (mean across batch, user can use `torchtraining.savers.Mean`
        to get mean across iterations/epochs).
    """

    def __init__(self, p: float = 2.0, eps: float = 1e-06, reduction=torch.mean):
        super().__init__()
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, data):
        """
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
        return functional.metrics.distance.pairwise(
            *data, self.p, self.eps, self.reduction,
        )
