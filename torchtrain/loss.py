# DICE, Focal Loss, Quadruplet Loss, Label Smoothing


import torch
from torch.nn.modules.loss import _WeightedLoss

from . import functional


class BinaryFocalLossWithLogits(_WeightedLoss):
    """Binary focal loss working with raw output from network (logits).

    See: https://arxiv.org/abs/1708.02002.

    Underplays loss of easy examples while leaving loss of harder examples
    for neural network mostly intact (dampened way less).

    The higher the gamma parameter, the greater the "focusing" effect.

    Arguments
    ---------
    gamma: float
        Scale of focal loss effect. To obtain binary crossentropy set it to 0.0.
        `0.5 - 2.5` range was used in original research paper and seemed robust.
    weight: Tensor, optional
        Manual rescaling weight, if provided it's repeated to match input
        tensor shape
    reduction: str, optional
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    alpha: Tensor, optional
        Weight of positive examples. Must be a vector with
        length equal to the number of classes.
        In general `alpha` should be decreased slightly as `gamma` is increased
        (for `gamma=2`, `alpha=0.25` works best).

    Shape
    -----
    inputs:
        :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        Usually of shape :math:`(N, H, W)`, where :math:`H` is image height and :math:`W`
        is it's width.
    targets:
        :math:`(N, *)`, same shape as the input.
    output: scalar
        If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, otherwise same
        shape as input.

    """

    def __init__(
        self, gamma: float, weight=None, pos_weight=None, reduction: str = "mean",
    ):
        super().__init__(weight, reduction=reduction)

        self.gamma = gamma
        self.pos_weight = pos_weight

    def __call__(self, inputs, targets):
        return functional.loss.binary_focal_loss_with_logits(
            inputs, targets, self.gamma, self.weight, self.reduction, self.pos_weights,
        )


class MulticlassFocalLoss(_WeightedLoss):
    """Multiclass focal loss working with raw output from network (logits).

    See: https://arxiv.org/abs/1708.02002.

    Underplays loss of easy examples while leaving loss of harder examples
    for neural network mostly intact (dampened way less).

    The higher the gamma parameter, the greater the "focusing" effect.

    Arguments
    ---------
    gamma: float
        Scale of focal loss effect. To obtain binary crossentropy set it to 0.0.
        `0.5 - 2.5` range was used in original research paper and seemed robust.
    weight: Tensor, optional
        Manual rescaling weight, if provided it's repeated to match input
        tensor shape.
    ignore_index int, optional
        Specifies a target value that is ignored and does not contribute to the input gradient.
        When :attr:`size_average` is ``True``, the loss is averaged over non-ignored targets.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.

    Shape
    -----
    inputs:
        :math:`(N, C)` where `C = number of classes`, or
        :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
        in the case of `K`-dimensional loss.
        Usually of shape :math:`(N, C, H, W)`, where :math:`H` is image height and :math:`W`
        is it's width.
    targets:
        :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
        :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
        K-dimensional loss.
        Usually of shape :math:`(N, H, W)`, where :math:`H` is image height and :math:`W`
        is it's width and elements are of specified `C` classes.
    output:
        If :attr:`reduction` is ``'none'``, then the same size as the target:
        :math:`(N)`, or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
        of K-dimensional loss. if :attr:`reduction` is ``'mean'`` or ``'sum'``
        a scalar.

    """

    def __init__(
        self,
        gamma: float,
        weight=None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(weight, reduction=reduction)

        self.gamma = gamma
        self.ignore_index = ignore_index

    def __call__(self, inputs, targets):
        return functional.loss.multiclass_focal_loss_with_logits(
            inputs, targets, self.gamma, self.weight, self.ignore_index, self.reduction
        )


# Inspired by: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
# and fixed according to: https://arxiv.org/abs/1906.02629
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        alpha: float,
        weight=None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(weight, reduction=reduction)

        if not 0 <= alpha < 1:
            raise ValueError("smoothing alpha should be in [0, 1) range.")

        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        return functional.loss.smooth_cross_entropy_loss(
            inputs, targets, self.alpha, self.weight, self.ignore_index, self.reduction
        )
