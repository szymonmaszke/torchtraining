import torch


def dimensions_number_check(output, target):
    if len(output.shape) - 1 != len(target.shape):
        raise ValueError(
            "Got {} number of dimensions for output and {} for target. "
            "Output should have one dimension more than target.".format(
                len(output.shape), len(target.shape),
            )
        )


def shape_check(output, target):
    if output.shape != target.shape[:-1]:
        raise ValueError(
            "Got {} shape for output and {} for target. "
            "Shape should be equal except for last dimension of output.".format(
                output.shape, target.shape
            )
        )


def check(output, target):
    _dimensions_number_check(output, target)
    _shape_check(output, target)


def categorical(output, target):
    return torch.argmax(output, dim=-1), target.long()


def one_hot(output, target, num_classes):
    categorical_output, categorical_target = _categorical(output, target)
    return (
        torch.nn.functional.one_hot(output, num_classes),
        torch.nn.functional.one_hot(target, num_classes),
    )


def get_num_classes(num_classes, output, target):
    if num_classes == -1:
        return max(output.shape[-1], torch.max(target).item())
    if num_classes < 1:
        raise ValueError(
            "num_classes has to be either -1 or positive value, got {}".format(
                num_classes
            )
        )
    return num_classes
