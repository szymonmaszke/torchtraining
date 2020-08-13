import typing

import torch


def _reduce(loss, reduction, inputs):
    if reduction == "mean":
        return torch.sum(loss) / inputs.shape[0]
    if reduction == "sum":
        return torch.sum(loss)
    if reduction == "none":
        return loss
    raise ValueError("{} is not a valid value for reduction".format(reduction))


# N, H, W
def binary_focal_loss(
    inputs,
    targets,
    gamma: float,
    weight=None,
    pos_weight=None,
    reduction: str = "mean",
):
    """See `torchtrain.loss.BinaryFocalLossWithLogits`."""

    probabilities = (1 - torch.sigmoid(inputs)) ** gamma
    loss = probabilities * torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, weight, reduction="none", pos_weight=pos_weight
    )
    return _reduce(loss, reduction, inputs)


def multiclass_focal_loss(
    inputs,
    targets,
    gamma: float,
    weight=None,
    ignore_index=-100,
    reduction: str = "mean",
):
    """See `torchtrain.loss.MulticlassFocalLossWithLogits`."""

    inputs[:, ignore_index, ...] = 0
    probabilities = (1 - torch.nn.functional.softmax(inputs, dim=1)) ** gamma
    loss = probabilities * torch.nn.functional.cross_entropy(
        inputs, targets, weight, ignore_index=ignore_index, reduction="none"
    )

    return _reduce(loss, reduction, inputs)


def smooth_binary_cross_entropy(
    inputs,
    targets,
    alpha: float,
    weight=None,
    pos_weight=None,
    reduction: str = "mean",
):

    """See `torchtrain.loss.SmoothBinaryCrossEntropy`."""
    inputs *= (1 - alpha) + alpha / 2
    return torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, weight, pos_weight=pos_weight, reduction=reduction
    )


def smooth_cross_entropy(
    inputs,
    targets,
    alpha: float,
    weight=None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    """See `torchtrain.loss.SmoothCrossEntropy`."""
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1])
    one_hot_targets *= (1 - alpha) + alpha / inputs.shape[-1]
    one_hot_targets[..., ignore_index] = inputs[..., ignore_index]
    loss = -(one_hot_targets * torch.nn.functional.log_softmax(inputs, dim=-1))
    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return _reduce(loss, reduction, inputs)


def quadruplet(
    anchor,
    positive,
    negative,
    negative2,
    alpha1: float,
    alpha2: float,
    metric: typing.Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = torch.nn.functional.pairwise_distance,
    weight=None,
    reduction: str = "sum",
):
    """See `torchtrain.loss.Quadruplet`."""
    loss = torch.nn.functional.relu(
        metric(anchor, positive) ** 2 - metric(anchor, negative) ** 2 + alpha1
    )
    +torch.nn.functional.relu(
        metric(anchor, positive) ** 2 - metric(negative, negative2) ** 2 + alpha2
    )

    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    return _reduce(loss, reduction, anchor)
