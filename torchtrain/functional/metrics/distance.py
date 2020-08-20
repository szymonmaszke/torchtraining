import typing

import torch

from .. import utils


# assert 2 size
@utils.docs
def cosine(
    output: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-08,
    reduction=torch.mean,
) -> torch.Tensor:
    return reduction(
        1 - torch.nn.functional.cosine_similarity(output, target, dim, eps)
    )


@utils.docs
def pairwise(
    output: torch.Tensor,
    target: torch.Tensor,
    p: float = 2.0,
    eps: float = 1e-06,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor,] = torch.mean,
) -> torch.Tensor:
    return reduction(torch.nn.functional.pairwise_distance(output, target, p, eps))
