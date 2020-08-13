import typing

import torch


def _get_reduction(reduction, inputs):
    if reduction is None:
        return lambda loss: loss.sum() / inputs.shape[0]
    return reduction


def binary_focal_loss(
    inputs,
    targets,
    gamma: float,
    weight=None,
    pos_weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """See `torchtrain.loss.BinaryFocalLoss`."""

    reduce = _get_reduction(reduction, inputs)
    probabilities = (1 - torch.sigmoid(inputs)) ** gamma
    loss = probabilities * torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, weight, reduction="none", pos_weight=pos_weight
    )
    return reduce(loss)


def multiclass_focal_loss(
    inputs,
    targets,
    gamma: float,
    weight=None,
    ignore_index=-100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """See `torchtrain.loss.MulticlassFocalLoss`."""

    reduce = _get_reduction(reduction, inputs)

    inputs[:, ignore_index, ...] = 0
    probabilities = (1 - torch.nn.functional.softmax(inputs, dim=1)) ** gamma
    loss = probabilities * torch.nn.functional.cross_entropy(
        inputs, targets, weight, ignore_index=ignore_index, reduction="none"
    )

    return reduce(loss)


def smooth_binary_cross_entropy(
    inputs,
    targets,
    alpha: float,
    weight=None,
    pos_weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """See `torchtrain.loss.SmoothBinaryCrossEntropy`."""
    reduce = _get_reduction(reduction, inputs)

    inputs *= (1 - alpha) + alpha / 2
    return reduce(torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, weight, pos_weight=pos_weight, reduction="none"
    ))


def smooth_cross_entropy(
    inputs,
    targets,
    alpha: float,
    weight=None,
    ignore_index: int = -100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """See `torchtrain.loss.SmoothCrossEntropy`."""

    reduce = _get_reduction(reduction, inputs)

    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1])
    one_hot_targets *= (1 - alpha) + alpha / inputs.shape[-1]
    one_hot_targets[..., ignore_index] = inputs[..., ignore_index]
    loss = -(one_hot_targets * torch.nn.functional.log_softmax(inputs, dim=-1))
    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return reduce(loss)


def quadruplet(
    anchor,
    positive,
    negative,
    negative2,
    alpha1: float = 1.0,
    alpha2: float = 0.5,
    metric: typing.Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = torch.nn.functional.pairwise_distance,
    weight=None,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """See `torchtrain.loss.Quadruplet`."""

    reduce = _get_reduction(reduction, inputs)

    loss = torch.nn.functional.relu(
        metric(anchor, positive) ** 2 - metric(anchor, negative) ** 2 + alpha1
    )
    +torch.nn.functional.relu(
        metric(anchor, positive) ** 2 - metric(negative, negative2) ** 2 + alpha2
    )

    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return reduce(loss)
