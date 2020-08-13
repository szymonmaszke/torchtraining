import typing

import torch


def mixup(
    inputs: torch.Tensor, targets: torch.Tensor, gamma: float
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Perform per-batch mixup on images.

    See [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
    for explanation of the method.
    """
    perm = torch.randperm(inputs.shape[0])
    perm_inputs = inputs[perm]
    perm_targets = targets[perm]
    return (
        inputs.mul_(gamma).add_(perm_inputs, alpha=1 - gamma),
        targets.mul_(gamma).add_(perm_targets, alpha=1 - gamma),
    )
