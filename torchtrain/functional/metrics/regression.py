import typing

import torch

from .. import _base


def total_of_squares(
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.sum,
) -> torch.Tensor:
    return reduction(target - torch.mean(target)) ** 2


def regression_of_squares(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.sum,
) -> torch.Tensor:
    return reduction(output - torch.mean(target)) ** 2


def squares_of_residuals(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.sum,
) -> torch.Tensor:
    return reduction(output - target) ** 2


# TBD
def r2(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    return 1 - squares_of_residuals(output, target) / total_of_squares(target)


def adjusted_r2(output: torch.Tensor, target: torch.Tensor, p: int) -> torch.Tensor:
    numel = output.numel()
    return 1 - (1 - r2(output, target)) * ((numel - 1) / (numel - p - 1))


def absolute_error(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.mean,
) -> torch.Tensor:
    return reduction(torch.nn.functional.l1_loss(output, target, reduction="none"))


def squared_error(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.mean,
) -> torch.Tensor:
    return reduction(torch.nn.functional.mse_loss(output, target, reduction="none"))


def squared_log_error(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.mean,
) -> torch.Tensor:
    return reduction((torch.log(1 + target) - torch.log(1 + output)) ** 2)


def max_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(output - target))
