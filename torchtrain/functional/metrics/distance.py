import typing

import torch

from . import utils


# assert 2 size
@utils.docs
def cosine(
    output: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    return 1 - (
        output
        @ target.T
        / torch.max(
            torch.dot(
                torch.nn.functional.norm(output, p=2, dim=0),
                torch.nn.functional.norm(target, p=2, dim=0),
            ),
            epsilon,
        )
    )


@utils.docs
def euclidean(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(output @ output - 2 * output @ target + target @ target)


@utils.docs
def pairwise(
    output: torch.Tensor,
    target: torch.Tensor,
    p: float = 2.0,
    eps: float = 1e-06,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.mean,
) -> torch.Tensor:
    return reduction(torch.nn.functional.pairwise_distance(output, target, p, eps))
