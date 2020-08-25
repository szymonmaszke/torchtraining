import itertools

import torch

import pytest
import torchtraining as tt


@pytest.mark.parametrize(
    "cast,inputs",
    itertools.product(
        (
            (tt.device.CPU(), torch.device("cpu")),
            (tt.device.Device(device=torch.device("cpu")), torch.device("cpu")),
        ),
        (torch.randn(4, 5), torch.randn(5)),
    ),
)
def test_cast(cast, inputs):
    caster, desired = cast
    assert caster(inputs).device == desired
