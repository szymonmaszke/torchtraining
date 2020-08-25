"""Utilities used with most binary metrics."""

import typing

import torch


def check(output: torch.Tensor, target: torch.Tensor) -> None:
    """
    Check `output` and `target` shape is equal.

    ValueError is raised otherwise

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target

    """

    if output.shape != target.shape:
        raise ValueError(
            "Output and target has to be of the same shape! Got {} for output and {} for target".format(
                output.shape, target.shape
            )
        )


def threshold(
    output: torch.Tensor, target: torch.Tensor, threshold: float
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Threshold output logits/probabilities at `thresholds`.

    For logits, `threshold` should be set to `0`, while for `probabilities`
    it should be `0.5`

    Parameters
    ----------
    output : torch.Tensor
        Usually neural network output
    target : torch.Tensor
        Usually desired target
    threshold : float
        Value at which `output`will be thresholded

    Returns
    -------
    typing.Tuple[torch.Tensor, torch.Tensor]:
        Thresholded output and `target` being a boolean for faster computations.
    """
    return output > threshold, target.bool()
