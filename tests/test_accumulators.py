import torch

import pytest
import torchtrain as tt


@pytest.mark.parametrize("inputs")
def test_sum(inputs, expected):
    accumulator = tt.accumulators.Sum()
    pass


@pytest.mark.parametrize("inputs")
def test_mean(inputs, expected):
    accumulator = tt.accumulators.Mean()
    pass


@pytest.mark.parametrize("inputs")
def test_list(inputs, expected):
    accumulator = tt.accumulators.List()
    pass


@pytest.mark.parametrize("obj,inputs,expected")
def test_except(obj, inputs, expected):
    pass
