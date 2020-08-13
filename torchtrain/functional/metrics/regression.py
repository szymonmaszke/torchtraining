import torch

from .. import _base


def total_of_squares(target, reduction=torch.sum):
    return reduction(target - torch.mean(target)) ** 2


def regression_of_squares(output, target, reduction=torch.sum):
    return reduction(output - torch.mean(target)) ** 2


def squares_of_residuals(output, target, reduction=torch.sum):
    return reduction(output - target) ** 2


# TBD
def r2(output, target):
    return 1 - squares_of_residuals(output, target) / total_of_squares(target)


def adjusted_r2(output, target, p):
    numel = output.numel()
    return 1 - (1 - r2(output, target)) * ((numel - 1) / (numel - p - 1))


def absolute_error(output, target, reduction=torch.mean):
    return reduction(torch.nn.functional.l1_loss(output, target, reduction="none"))


def squared_error(output, target, reduction=torch.mean):
    return reduction(torch.nn.functional.mse_loss(output, target, reduction="none"))


def squared_log_error(output, target, reduction=torch.mean):
    return reduction((torch.log(1 + target) - torch.log(1 + output)) ** 2)


def max_error(output, target):
    return torch.max(torch.abs(output - target))
