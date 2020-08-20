import typing

import torch

from . import utils


def _to_tensor(value):
    if value is None or torch.is_tensor(value):
        return value
    return torch.tensor(value)


def _get_reduction(reduction):
    if reduction is None:
        return lambda loss: loss.sum() / loss.shape[0]
    return reduction


@utils.docs
def binary_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight=None,
    pos_weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)
    weight, pos_weight = _to_tensor(weight), _to_tensor(pos_weight)

    probabilities = (1 - torch.sigmoid(outputs)) ** gamma
    loss = probabilities * torch.nn.functional.binary_cross_entropy_with_logits(
        outputs, targets.float(), weight, reduction="none", pos_weight=pos_weight,
    )

    return reduce(loss)


@utils.docs
def multiclass_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight=None,
    ignore_index=-100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    if ignore_index >= 0:
        outputs[:, ignore_index, ...] = 0
    probabilities = (1 - torch.nn.functional.softmax(outputs, dim=1)) ** gamma
    loss = probabilities * torch.nn.functional.cross_entropy(
        outputs, targets, weight, ignore_index=ignore_index, reduction="none"
    ).unsqueeze(dim=1)

    return reduce(loss)


@utils.docs
def smooth_binary_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    weight=None,
    pos_weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    weight, pos_weight = _to_tensor(weight), _to_tensor(pos_weight)
    targets = targets * (1 - alpha) + alpha / 2

    return reduce(
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs, targets, weight, pos_weight=pos_weight, reduction="none"
        )
    )


@utils.docs
def smooth_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    weight=None,
    ignore_index: int = -100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:

    reduce = _get_reduction(reduction)

    # All classes may not occur in loss, specify num_classes explicitly
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1])
    permutation = list(range(len(one_hot_targets.shape)))
    permutation.insert(1, permutation[-1])

    # We leave last dimension as it's inserted at correct 1 position
    one_hot_targets = one_hot_targets.permute(permutation[:-1])

    smoothed_targets = one_hot_targets * (1 - alpha) + alpha / outputs.shape[1]
    if ignore_index >= 0:
        smoothed_targets[:, ignore_index, ...] = outputs[:, ignore_index, ...]
    log_softmax = torch.nn.functional.log_softmax(outputs, dim=1).unsqueeze(dim=1)
    loss = -(smoothed_targets * log_softmax)
    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return reduce(loss)


@utils.docs
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
