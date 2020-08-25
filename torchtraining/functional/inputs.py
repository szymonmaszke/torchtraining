"""
This module has no object oriented counterpart and should be used
in a functional fashion inside `forward` of `torchtraining.steps` (as the only one currently).

See `examples` for specific user cases

"""

import typing

import torch


def mixup(
    inputs: torch.Tensor, targets: torch.Tensor, gamma: float
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Perform per-batch mixup on images.

    See `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`__.
    for explanation of the method.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                images, labels = sample
                images, labels = tt.functional.inputs.mixup(images, labels)
                # Calculate what you want below, say loss
                ...
                return loss


        step = TrainStep(criterion, device)

    .. note::

        **IMPORTANT**: Examples are modified in-place!

    Arguments
    ---------
    inputs: torch.Tensor
        `torch.Tensor` of shape :math:`(N, *)` and numerical `dtype`.
    labels: torch.Tensor
        `torch.Tensor` of shape :math:`(N, *)` and numerical `dtype`.
    gamma: float
        Level of mixing between inputs and labels. The smaller the value,
        the more "concrete" examples are (e.g. for  `0.1` and `cat`, `dog` labels
        it would be `0.9` cat and `0.1` dog).

    Returns
    -------
    Tuple(torch.Tensor, torch.Tensor)
        Inputs and labels after mixup (linear mix with `gamma` strength).

    """
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            "inputs and labels 0 dimension (batch) has to be equal, "
            "got {} for inputs and {} for labels".format(
                inputs.shape[0], targets.shape[0]
            )
        )
    perm = torch.randperm(inputs.shape[0])
    perm_inputs = inputs[perm]
    perm_targets = targets[perm]
    return (
        inputs.mul_(gamma).add_(perm_inputs, alpha=1 - gamma),
        targets.mul_(gamma).add_(perm_targets, alpha=1 - gamma),
    )
