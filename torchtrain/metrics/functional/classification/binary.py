import typing

import torch

from . import utils

# Useful interfaces


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable = torch.mean,
    threshold: float = 0.0,
):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return reduction((output == target).float())


def jaccard(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    threshold: float = 0.0,
):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)

    union = (output | target).sum(axis=-1)
    intersection = (target & output).sum(axis=-1)
    empty = union <= 0
    union[empty] = 1
    intersection[empty] = 1

    return reduction(intersection.float() / union)


# Basic cases


def true_positive(output, target, reduction=torch.sum, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return reduction((output & target).float())


def false_positive(output, target, reduction=torch.sum, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return reduction((output & ~target).float())


def true_negative(output, target, reduction=torch.sum, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return reduction((~output & ~target).float())


def false_negative(output, target, reduction=torch.sum, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return reduction((~output & target).float())


# Confusion matrix


def confusion_matrix(output, target, reduction=torch.sum, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    tp = reduction((output & target).float())
    fp = reduction((output & ~target).float())
    tn = reduction((~output & ~target).float())
    fn = reduction((~output & target).float())
    return torch.tensor([tp, fn, fp, tn]).reshape(2, 2)


# Rate metrics


def recall(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return (output & target).sum().float() / target.sum()


def specificity(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    inverse_target = ~target
    return (~output & inverse_target).sum().float() / inverse_target.sum()


def precision(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return (output & target).sum().float() / output.sum()


def negative_predictive_value(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    inverse_output = ~output
    return (inverse_output & ~target).sum().float() / inverse_output.sum()


def false_negative_rate(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return (~output & target).sum().float() / target.sum()


def false_positive_rate(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    inverse_target = ~target
    return (output & inverse_target).sum().float() / inverse_target.sum()


def false_discovery_rate(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    return (output & ~target).sum().float() / output.sum()


def false_omission_rate(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    inverse_output = ~output
    return (inverse_output & target).sum().float() / inverse_output.sum()


# Other related to above metrics

# Like F1-score almost
def critical_success_index(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)
    tp = (output & target).sum().float()
    return tp / tp + (output != target).sum()


def balanced_accuracy(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)

    inverse_target = ~target
    return (
        (output & target).sum().float() / target.sum()
        + (~output & inverse_target).sum().float() / inverse_target.sum()
    ) / 2


def f1(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)

    tp = 2 * (output & target).sum().float()
    return tp / (tp + (output != target).sum())


def fbeta(output, target, beta: float, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)

    tp = (1 + beta) ** 2 * (output & target).sum().float()
    return tp / (tp + (beta ** 2) * (output != target).sum())


def matthews_correlation_coefficient(output, target, threshold: float = 0.0):
    utils.binary.check(output, target)
    output, target = utils.binary.threshold(output, target, threshold)

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
