import torch


def mixup(inputs, labels, gamma: float):
    """Perform per-batch mixup on images.

    See [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
    for explanation of the method.
    """
    perm = torch.randperm(images.size(0))
    perm_input = images[perm]
    perm_target = labels[perm]
    return (
        images.mul_(gamma).add_(perm_input, alpha=1 - gamma),
        labels.mul_(gamma).add_(perm_target, alpha=1 - gamma),
    )
