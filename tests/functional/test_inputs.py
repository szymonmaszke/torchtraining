import itertools

import torch

import pytest
import torchtrain as tt


@pytest.mark.parametrize(
    "inputs,gamma",
    itertools.product(
        list(
            itertools.permutations(
                (torch.ones(64, 1), torch.randn(32, 3, 28, 28), torch.randn(32, 64)),
                r=2,
            )
        ),
        (0.1, 1.0, 0.0),
    ),
)
def test_mixup(inputs, gamma):
    inputs, labels = inputs
    expected = gamma != 0

    copy_inputs, copy_labels = inputs.copy(), labels.copy()
    mixed_inputs, mixed_labels = tt.functional.inputs.mixup(inputs, labels)
    assert (copy_inputs != mixed_inputs) == expected
    assert (copy_labels != mixed_labels) == expected
