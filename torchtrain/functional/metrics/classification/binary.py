import typing

import torch

from .. import utils
from . import utils as binary_utils


@utils.docstring
def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return reduction((output == target).float())


@utils.docstring
def jaccard(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    union = (output | target).sum(axis=-1)
    intersection = (target & output).sum(axis=-1)
    empty = union <= 0
    union[empty] = 1
    intersection[empty] = 1

    return reduction(intersection.float() / union)


# Basic cases


@utils.docstring
def true_positive(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return reduction((output & target).float())


@utils.docstring
def false_positive(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return reduction((output & ~target).float())


@utils.docstring
def true_negative(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return reduction((~output & ~target).float())


@utils.docstring
def false_negative(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return reduction((~output & target).float())


# Confusion matrix


@utils.docstring
def confusion_matrix(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    tp = reduction((output & target).float())
    fp = reduction((output & ~target).float())
    tn = reduction((~output & ~target).float())
    fn = reduction((~output & target).float())

    return torch.tensor([tp, fn, fp, tn]).reshape(2, 2)


# Rate metrics


@utils.docstring
def recall(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return (output & target).sum().float() / target.sum()


@utils.docstring
def specificity(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)
    inverse_target = ~target

    return (~output & inverse_target).sum().float() / inverse_target.sum()


@utils.docstring
def precision(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return (output & target).sum().float() / output.sum()


@utils.docstring
def negative_predictive_value(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)
    inverse_output = ~output

    return (inverse_output & ~target).sum().float() / inverse_output.sum()


@utils.docstring
def false_negative_rate(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return (~output & target).sum().float() / target.sum()


@utils.docstring
def false_positive_rate(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)
    inverse_target = ~target

    return (output & inverse_target).sum().float() / inverse_target.sum()


@utils.docstring
def false_discovery_rate(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    return (output & ~target).sum().float() / output.sum()


@utils.docstring
def false_omission_rate(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)
    inverse_output = ~output

    return (inverse_output & target).sum().float() / inverse_output.sum()


@utils.docstring
def critical_success_index(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)
    tp = (output & target).sum().float()

    return tp / tp + (output != target).sum()


@utils.docstring
def balanced_accuracy(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)
    inverse_target = ~target

    return (
        (output & target).sum().float() / target.sum()
        + (~output & inverse_target).sum().float() / inverse_target.sum()
    ) / 2


@utils.docstring
def f1(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    tp = 2 * (output & target).sum().float()

    return tp / (tp + (output != target).sum())


@utils.docstring
def f_beta(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    tp = (1 + beta) ** 2 * (output & target).sum().float()

    return tp / (tp + (beta ** 2) * (output != target).sum())


@utils.docstring
def matthews_correlation_coefficient(
    output: torch.Tensor, target: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    binary_utils.binary.check(output, target)
    output, target = binary_utils.binary.threshold(output, target, threshold)

    inverse_output = ~output
    inverse_target = ~target

    tp = (output & target).float()
    tn = (inverse_output & inverse_target).float()
    fp = (output & inverse_target).float()
    fn = (inverse_output & target).float()

    numerator = torch.dot(tp, tn) - torch.dot(fp, fn)
    denominator = (
        output.sum() * target.sum() * inverse_target.sum() * inverse_output.sum()
    )

    if denominator == 0.0:
        return numerator
    return numerator / denominator
