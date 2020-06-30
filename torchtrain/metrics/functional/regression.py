import torch

from .. import _base


# TBD
def r2_score():
    pass


def absolute_error(output, target, reduction=torch.mean):
    return reduction(torch.nn.functional.l1_loss(output, target, reduction="none"))


def squared_error(output, target, reduction=torch.mean):
    return reduction(torch.nn.functional.mse_loss(output, target, reduction="none"))


def pairwise_distance(
    output, target, p: float = 2.0, eps: float = 1e-06, reduction=torch.mean
):
    return reduction(torch.nn.functional.pairwise_distance(output, target, p, eps))
