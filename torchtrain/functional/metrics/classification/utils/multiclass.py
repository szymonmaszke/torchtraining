import typing

import torch


def dimensions_number_check(output: torch.Tensor, target: torch.Tensor) -> None:
    """
    Validate output and target number of dimensions.

    Output should have one dimension more than target as those are probabilities
    or logits.

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target

    """
    if len(output.shape) - 1 != len(target.shape):
        raise ValueError(
            "Got {} number of dimensions for output and {} for target. "
            "Output should have one dimension more than target.".format(
                len(output.shape), len(target.shape),
            )
        )


def shape_check(output: torch.Tensor, target: torch.Tensor) -> None:
    """
    Validate output and target shapes (except last dimension).

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target

    """
    output_shape = list(output.shape)
    output_shape.pop(1)
    if output_shape != list(target.shape):
        raise ValueError(
            "Got {} shape for output and {} for target. "
            "Shape should be equal except for 1 dimension of output (specifying classes).".format(
                output.shape, target.shape
            )
        )


def check(output: torch.Tensor, target: torch.Tensor) -> None:
    """
    Run all multiclass checks.

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target

    """
    dimensions_number_check(output, target)
    shape_check(output, target)


def categorical(
    output: torch.Tensor, target: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform output and target to categorical.

    Additionally, `target` will be transformed to `long` for faster comparison
    (as it is already a categorical).

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target

    Returns
    -------
    typing.Tuple[torch.Tensor, torch.Tensor]:
        Categorical `output` and `target`.

    """
    return torch.argmax(output, dim=1).long(), target.long()


def one_hot(
    output: torch.Tensor, target: torch.Tensor, num_classes: int = -1
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform output and target to one hot encoding.

    Both will be transformed to categorical beforehand

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target

    Returns
    -------
    typing.Tuple[torch.Tensor, torch.Tensor]:
        One hot encoded `output` and `target`.

    """
    categorical_output, categorical_target = categorical(output, target)
    return (
        torch.nn.functional.one_hot(categorical_output, num_classes),
        torch.nn.functional.one_hot(categorical_target, num_classes),
    )
