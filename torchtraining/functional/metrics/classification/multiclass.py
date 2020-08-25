import typing

import torch

from ... import utils
from . import utils as multiclass_utils


@utils.docs
def topk(
    output: torch.Tensor,
    target: torch.Tensor,
    k: int,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)

    biggest_indices = torch.topk(output, k, dim=-1)[1]
    equal = target.expand(*(target.shape), k) == biggest_indices

    return reduction(equal.sum(dim=-1))


@utils.docs
def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.categorical(output, target)

    return reduction((output == target).float())


@utils.docs
def jaccard(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    union = (output | target).sum(axis=-1)
    intersection = (target & output).sum(axis=-1)
    empty = union <= 0
    union[empty] = 1
    intersection[empty] = 1

    return reduction(intersection.float() / union)


# Basic cases


@utils.docs
def true_positive(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return reduction((output & target).float(), dim=-1)


@utils.docs
def false_positive(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return reduction((output & ~target).float(), dim=-1)


@utils.docs
def true_negative(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return reduction((~output & ~target).float(), dim=-1)


@utils.docs
def false_negative(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor,], torch.Tensor] = torch.sum,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return reduction((~output & target).float(), dim=-1)


@utils.docs
def confusion_matrix(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    num_classes = output.shape[1]
    output, target = multiclass_utils.multiclass.categorical(output, target)

    unique_labels = (target * num_classes + output).flatten()

    return torch.bincount(unique_labels, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


@utils.docs
def recall(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return (output & target).sum().float() / target.sum()


@utils.docs
def specificity(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )
    inverse_target = ~target

    return (~output & inverse_target).sum().float() / inverse_target.sum()


@utils.docs
def precision(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return (output & target).sum().float() / output.sum()


@utils.docs
def negative_predictive_value(
    output: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )
    inverse_output = ~output

    return (inverse_output & ~target).sum().float() / inverse_output.sum()


@utils.docs
def false_negative_rate(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return (~output & target).sum().float() / target.sum()


@utils.docs
def false_positive_rate(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )
    inverse_target = ~target

    return (output & inverse_target).sum().float() / inverse_target.sum()


@utils.docs
def false_discovery_rate(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    return (output & ~target).sum().float() / output.sum()


@utils.docs
def false_omission_rate(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )
    inverse_output = ~output

    return (inverse_output & target).sum().float() / inverse_output.sum()


@utils.docs
def critical_success_index(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    tp = (output & target).sum().float()

    return tp / tp + (output != target).sum()


@utils.docs
def balanced_accuracy(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )
    inverse_target = ~target

    return (
        (output & target).sum().float() / target.sum()
        + (~output & inverse_target).sum().float() / inverse_target.sum()
    ) / 2


@utils.docs
def f1(output: torch.Tensor, target: torch.Tensor,) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    tp = 2 * (output & target).sum().float()

    return tp / (tp + (output != target).sum())


@utils.docs
def f_beta(output: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )

    tp = (1 + beta) ** 2 * (output & target).sum().float()

    return tp / (tp + (beta ** 2) * (output != target).sum())


@utils.docs
def matthews_correlation_coefficient(
    output: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    multiclass_utils.multiclass.check(output, target)
    output, target = multiclass_utils.multiclass.one_hot(
        output, target, num_classes=output.shape[1]
    )
    inverse_output = ~output
    inverse_target = ~target

    dimensions = tuple(range(len(inverse_output.shape)))

    tp = (output & target).float().sum(dim=dimensions[1:])
    tn = (inverse_output & inverse_target).float().sum(dim=dimensions[1:])
    fp = (output & inverse_target).float().sum(dim=dimensions[1:])
    fn = (inverse_output & target).float().sum(dim=dimensions[1:])

    numerator = torch.dot(tp, tn) - torch.dot(fp, fn)
    denominator = (
        output.sum() * target.sum() * inverse_target.sum() * inverse_output.sum()
    )

    if denominator == 0.0:
        return numerator
    return numerator / denominator
