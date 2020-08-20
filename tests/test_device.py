import itertools

import torch

import pytest
import torchfunc
import torchtrain as tt


@pytest.mark.parametrize(
    "cast,inputs",
    itertools.product(
        (
            (tt.device.CPU(), torch.device("cpu")),
            (tt.device.Device(device=torch.device("cpu")), torch.device("cpu")),
        ),
        (torch.nn.Linear(20, 10), torch.randn(4, 5)),
    ),
)
def test_cast(cast, inputs):
    caster, desired = cast
    if not isinstance(inputs, torch.nn.Module):
        assert cast(inputs).device == desired
    else:
        assert torchfunc.module.device(inputs) == desired
