import typing

import torch
from torch.nn.modules.loss import _Loss

from . import functional

# DICE


# Dice loss


class BinaryFocal(_Loss):
    """Binary focal loss working with raw output from network (logits).

    See original research paper: `Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`__

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
    pos_weight: Tensor, optional
        Weight of positive examples. Must be a vector with
        length equal to the number of classes.
        In general `pos_weight` should be decreased slightly as `gamma` is increased
        (for `gamma=2`, `pos_weight=0.25` was found to work best in original paper).
    reduction: typing.Callable(torch.Tensor) -> torch.Tensor, optional
        Specifies the reduction to apply to the output.
        If user wants no reduction he should use: `lambda loss: loss`.
        If user wants a summation he should use: `torch.sum`.
        By default, `lambda loss: loss.sum() / loss.shape[0]` is used (mean across examples).


    """

    def __init__(
        self,
        gamma: float,
        weight=None,
        pos_weight=None,
        reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()

        self.gamma = gamma
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        outputs: torch.Tensor
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
            Usually of shape :math:`(N, H, W)`, where :math:`H` is image height and :math:`W`
            is it's width.
        targets: torch.Tensor
            :math:`(N, *)`, same shape as the input.

        Returns
        -------
        torch.Tensor
            If :attr:`reduction` is not specified then `mean` across sample is taken.
            Otherwise whatever shape `reduction` returns.

        """
        return functional.loss.binary_focal_loss(
            outputs, targets, self.gamma, self.weight, self.pos_weight, self.reduction,
        )


class MulticlassFocal(_Loss):
    r"""Multiclass focal loss working with raw output from network (logits).

    See original research paper: `Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`__

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
    reduction: typing.Callable(torch.Tensor) -> torch.Tensor, optional
        Specifies the reduction to apply to the output.
        If user wants no reduction he should use: `lambda loss: loss`.
        If user wants a summation he should use: `torch.sum`.
        By default, `lambda loss: loss.sum() / loss.shape[0]` is used (mean across examples).

    """

    def __init__(
        self,
        gamma: float,
        weight=None,
        ignore_index: int = -100,
        reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()

        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        outputs: torch.Tensor
            :math:`(N, C)` where `C = number of classes`, or
            :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
            in the case of `K`-dimensional loss.
            Usually of shape :math:`(N, C, H, W)`, where :math:`H` is image height and :math:`W`
            is it's width.
        targets: torch.Tensor
            :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
            K-dimensional loss.
            Usually of shape :math:`(N, H, W)`, where :math:`H` is image height and :math:`W`
            is it's width and elements are of specified `C` classes.

        Returns
        -------
        torch.Tensor
            If :attr:`reduction` is not specified then `mean` across sample is taken.
            Otherwise whatever shape `reduction` returns.

        """
        return functional.loss.multiclass_focal_loss(
            outputs, targets, self.gamma, self.weight, self.ignore_index, self.reduction
        )


# Inspired by: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
# and adjusted to fit torchtraining API
class SmoothCrossEntropy(_Loss):
    r"""Run cross entropy with non-integer labels smoothed by `alpha`.

    See `When Does Label Smoothing Help? <https://arxiv.org/abs/1906.02629>`__ for more details

    `targets` will be transformed to one-hot encoding and modified according
    to formula:

    .. math::
        y = y(1 - \alpha) + \frac{\alpha}{C}

    where :math:`C` is total number of classes.

    Arguments
    ---------
    alpha: float
        Smoothing parameter in the range `[0, 1)`.
    weight: Tensor, optional
        Manual rescaling weight, if provided it's repeated to match input
        tensor shape. Default: `None` (no weighting)
    ignore_index int, optional
        Specifies a target value that is ignored and does not contribute to the input gradient.
        When :attr:`size_average` is ``True``, the loss is averaged over non-ignored targets.
        Default: `-100`
    reduction: typing.Callable(torch.Tensor) -> torch.Tensor, optional
        Specifies the reduction to apply to the output.
        If user wants no reduction he should use: `lambda loss: loss`.
        If user wants a summation he should use: `torch.sum`.
        By default, `lambda loss: loss.sum() / loss.shape[0]` is used (mean across examples).

    """

    def __init__(
        self,
        alpha: float,
        weight=None,
        ignore_index: int = -100,
        reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        if not 0 <= alpha < 1:
            raise ValueError("smoothing alpha should be in [0, 1) range.")

        super().__init__()

        self.alpha = alpha
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        outputs: torch.Tensor
            :math:`(N, C)` where `C = number of classes`, or
            :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
            in the case of `K`-dimensional loss.
        targets: torch.Tensor
            :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
            K-dimensional loss.
        Returns
        -------
        torch.Tensor
            If :attr:`reduction` is not specified then `mean` across sample is taken.
            Otherwise whatever shape `reduction` returns.

        """
        return functional.loss.smooth_cross_entropy(
            outputs, targets, self.alpha, self.weight, self.ignore_index, self.reduction
        )


class SmoothBinaryCrossEntropy(_Loss):
    r"""Run binary cross entropy with booleans smoothed by `alpha`.

    See `When Does Label Smoothing Help? <https://arxiv.org/abs/1906.02629>`__ for more details

    `targets` will be transformed to one-hot encoding and modified according
    to formula:

    .. math::
        y = y(1 - \alpha) + \frac{\alpha}{2}

    where :math:`2` is total number of classes in binary case.

    Arguments
    ---------
    alpha: float
        Smoothing parameter in the range `[0, 1)`.
    weight: Tensor, optional
        Manual rescaling weight, if provided it's repeated to match input
        tensor shape. Default: `None` (no weighting)
    pos_weight: Tensor, optional
        Weight of positive examples. Must be a vector with
        length equal to the number of classes.
        In general `pos_weight` should be decreased slightly as `gamma` is increased
        (for `gamma=2`, `pos_weight=0.25` was found to work best in original paper).
    reduction: typing.Callable(torch.Tensor) -> torch.Tensor, optional
        Specifies the reduction to apply to the output.
        If user wants no reduction he should use: `lambda loss: loss`.
        If user wants a summation he should use: `torch.sum`.
        By default, `lambda loss: loss.sum() / loss.shape[0]` is used (mean across examples).


    """

    def __init__(
        self,
        alpha: float,
        weight=None,
        pos_weight: int = None,
        reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        if not 0 <= alpha < 1:
            raise ValueError("smoothing alpha should be in [0, 1) range.")

        super().__init__()

        self.alpha = alpha
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        outputs: torch.Tensor
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        targets: torch.Tensor
            :math:`(N, *)`, same shape as the input

        Returns
        -------
        torch.Tensor
            If :attr:`reduction` is not specified then `mean` across sample is taken.
            Otherwise whatever shape `reduction` returns.
        """
        return functional.loss.smooth_binary_cross_entropy(
            outputs, targets, self.alpha, self.weight, self.pos_weight, self.reduction
        )


class QuadrupletLoss(_Loss):
    r"""Quadruplet loss pushing away samples belonging to different classes.

    See original research paper `Beyond triplet loss: a deep quadruplet network for person re-identification <https://arxiv.org/abs/1704.01719>`__ for more information.

    It is an extension of `torch.nn.TripletMarginLoss`, where samples
    from two different `negative` (`negative` and `negative2`)
    classes should be pushed further away
    in space than those belonging to the same class (`anchor` and `positive`)

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a, p) - d(a, n) + {\rm alpha1}, 0\} + \max \{d(a, p) - d(n, n2) + {\rm alpha2}, 0\}

    Arguments
    ---------
    alpha1: float, optional
        Margin of standard `triplet` loss. Default: `1.0`
    alpha2: float, optional
        Margin of second part of loss (pushing negative1 and negative2 samples
        more than positive and anchor). Default: `0.5`
    metric: Callable(torch.Tensor, torch.Tensor) -> torch.Tensor, optional
        Metric used to rate distance between samples. Fully Connected neural
        network with one output and `sigmoid` could be used (as in original paper)
        or anything else adhering to API.
        Default: Euclidean distance.
    weight: Tensor, optional
        Manual rescaling weight, if provided it's repeated to match input
        tensor shape. Default: `None` (no weighting)
    reduction: typing.Callable(torch.Tensor) -> torch.Tensor, optional
        Specifies the reduction to apply to the output.
        If user wants no reduction he should use: `lambda loss: loss`.
        If user wants a summation he should use: `torch.sum`.
        By default, `lambda loss: loss.sum() / loss.shape[0]` is used (mean across examples).


    """

    def __init__(
        self,
        alpha1: float = 1.0,
        alpha2: float = 0.5,
        metric: typing.Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = torch.nn.functional.pairwise_distance,
        weight=None,
        reduction: str = "sum",
    ):
        super().__init__()

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric

        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        negative2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        anchor: torch.Tensor
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions
            For images usually of shape :math:`(N, C, H, W)`.
        positive: torch.Tensor
            Same as `anchor`
        negative: torch.Tensor
            Same as `anchor`
        negative2: torch.Tensor
            Same as `anchor`

        Returns
        -------
        torch.Tensor
            If :attr:`reduction` is not specified then `mean` across sample is taken.
            Otherwise whatever shape `reduction` returns.
        """
        return functional.loss.quadruplet(
            anchor,
            positive,
            negative,
            negative2,
            self.alpha1,
            self.alpha2,
            self.metric,
            self.weight,
            self.reduction,
        )
