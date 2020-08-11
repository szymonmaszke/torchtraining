import torch


# N, H, W
def binary_focal_loss_with_logits(
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
    if reduction == "mean":
        return torch.sum(loss) / inputs.shape[0]
    if reduction == "sum":
        return torch.sum(loss)
    return loss


def multiclass_focal_loss_with_logits(
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
    if reduction == "mean":
        return torch.sum(loss) / inputs.shape[0]
    if reduction == "sum":
        return torch.sum(loss)
    return loss


def smooth_cross_entropy_loss(
    inputs,
    targets,
    alpha: float,
    weight=None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1])
    one_hot_targets *= (1 - alpha) + alpha / inputs.shape[-1]
    one_hot_targets[..., ignore_index] = inputs[..., ignore_index]
    loss = -(one_hot_targets * torch.nn.functional.log_softmax(inputs, dim=-1))
    if weight is not None:
        loss *= weight.unsqueeze(dim=0)

    if reduction == "mean":
        return loss / inputs.shape[0]
    if reduction == "sum":
        return loss.sum()
    return loss
