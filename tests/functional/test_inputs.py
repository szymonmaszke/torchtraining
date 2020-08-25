import itertools

import torch

import pytest
import torchtraining as tt


@pytest.mark.parametrize(
    "inputs,gamma",
    itertools.product(
        list(
            itertools.permutations(
                (torch.randn(64), torch.randn(64, 3, 28, 28), torch.randn(64, 64)), r=2,
            )
        ),
        (0.0, 1.0, 0.2, 0.5),
    ),
)
def test_mixup(inputs, gamma):
    inputs, labels = inputs
    expected = gamma == 1

    copy_inputs, copy_labels = inputs.clone(), labels.clone()
    mixed_inputs, mixed_labels = tt.functional.inputs.mixup(inputs, labels, gamma)
    assert (copy_inputs == mixed_inputs).all() == expected
    assert (copy_labels == mixed_labels).all() == expected
