# import torch

# import pytest
# import torchtraining as tt


# @pytest.mark.parametrize("inputs", [range(5), range(0), range(20)])
# def test_sum(inputs):
#     accumulator = tt.accumulators.Sum()
#     for i in inputs:
#         accumulator(i)
#     assert accumulator.calculate() == sum(inputs)


# @pytest.mark.parametrize("inputs", [range(5), range(1), range(20)])
# def test_mean(inputs):
#     accumulator = tt.accumulators.Mean()
#     for i in inputs:
#         accumulator(i)
#     assert accumulator.calculate() == sum(inputs) / len(inputs)


# @pytest.mark.parametrize("inputs", [range(5), range(0), range(20)])
# def test_list(inputs):
#     accumulator = tt.accumulators.List()
#     for i in inputs:
#         accumulator(i)
#     assert accumulator.calculate() == list(inputs)
