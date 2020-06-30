import typing

import torch

# Useful interfaces

# Precision, Recall, ConfusionMatrix, AbsoluteError, SquaredError, PairwiseDistance,
# TopK, F2, FBeta, MCC, TruePositive, FalsePositive, TrueNegative, FalseNegative,
# AreaUnderCurve, Dice


def _output_target_same_shape(output, target):
    if output.shape != target.shape:
        raise ValueError(
            "Output and target has to be of the same shape! Got {} for output and {} for target".format(
                output.shape, target.shape
            )
        )


def _prepare_data(output, target, threshold):
    _output_target_same_shape(output, target)

    output = output > threshold
    target = target.bool()
    return output, target


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable = torch.mean,
    threshold: float = 0.0,
):
    output, target = _prepare_data(output, target, threshold)
    return reduction((output == target).float())


def jaccard(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    threshold: float = 0.0,
):
    output, target = _prepare_data(output, target, threshold)

    union = (output | target).sum(axis=-1)
    intersection = (target & output).sum(axis=-1)
    empty = union <= 0
    union[empty] = 1
    intersection[empty] = 1

    return reduction(intersection.float() / union)


def true_positive(output, target, reduction=torch.sum, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return reduction((output & target).float())


def false_positive(output, target, reduction=torch.sum, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return reduction((output & ~target).float())


def true_negative(output, target, reduction=torch.sum, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return reduction((~output & ~target).float())


def false_negative(output, target, reduction=torch.sum, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return reduction((~output & target).float())


def confusion_matrix(output, target, reduction=torch.sum, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    tp = reduction((output & target).float())
    fp = reduction((output & ~target).float())
    tn = reduction((~output & ~target).float())
    fn = reduction((~output & target).float())
    return torch.tensor([tp, fn, fp, tn]).view(2, 2)


def recall(output, target, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return (output & target).sum().float() / target.sum()


def specificity(output, target, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return (~output & ~target).sum().float() / (~target).sum()


def precision(output, target, threshold: float = 0.0):
    output, target = _prepare_data(output, target, threshold)
    return (output & target).sum().float() / output.sum()


# TBD Ignite source
def topk(output, target, k: int, reduction=torch.sum, threshold: float = 0.0):
    indices = torch.topk(output, k, dim=1)[1]
    expanded_y = y.view(-1, 1).expand(-1, k)
    correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
    self._num_correct += torch.sum(correct).item()
    self._num_examples += correct.shape[0]
    pass


def area_under_curve(output, target, reduction=torch.sum, threshold: float = 0.0):
    pass


def f2(output, target, reduction=torch.sum, threshold: float = 0.0):
    pass


def fbeta(output, target, reduction=torch.sum, threshold: float = 0.0):
    pass


def matthews_correlation_coefficient(
    output, target, reduction=torch.sum, threshold: float = 0.0
):
    pass


def dice(output, target, reduction=torch.sum, threshold: float = 0.0):
    pass
