import typing

import torch

from . import utils


def _get_reduction(reduction):
    if reduction is None:
        return lambda loss: loss.sum() / loss.shape[0]
    return reduction


@utils.docstring
def binary_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight=None,
    pos_weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    probabilities = (1 - torch.sigmoid(inputs)) ** gamma
    loss = probabilities * torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, weight, reduction="none", pos_weight=pos_weight
    )

    return reduce(loss)


@utils.docstring
def multiclass_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight=None,
    ignore_index=-100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    inputs[:, ignore_index, ...] = 0
    probabilities = (1 - torch.nn.functional.softmax(inputs, dim=1)) ** gamma
    loss = probabilities * torch.nn.functional.cross_entropy(
        inputs, targets, weight, ignore_index=ignore_index, reduction="none"
    )

    return reduce(loss)


@utils.docstring
def smooth_binary_cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    weight=None,
    pos_weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    inputs *= (1 - alpha) + alpha / 2

    return reduce(
        torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, weight, pos_weight=pos_weight, reduction="none"
        )
    )


@utils.docstring
def smooth_cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    weight=None,
    ignore_index: int = -100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1])
    one_hot_targets *= (1 - alpha) + alpha / inputs.shape[-1]
    one_hot_targets[..., ignore_index] = inputs[..., ignore_index]
    loss = -(one_hot_targets * torch.nn.functional.log_softmax(inputs, dim=-1))
    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return reduce(loss)


@utils.docstring
def quadruplet(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    negative2: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.5,
    metric: typing.Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = torch.nn.functional.pairwise_distance,
    weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    loss = torch.nn.functional.relu(
        metric(anchor, positive) ** 2 - metric(anchor, negative) ** 2 + alpha1
    )
    +torch.nn.functional.relu(
        metric(anchor, positive) ** 2 - metric(negative, negative2) ** 2 + alpha2
    )

    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return reduce(loss)
