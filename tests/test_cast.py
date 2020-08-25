import itertools

import pytest
import torch
import torchtraining as tt


class Check(tt.Operation):
    def __init__(self, desired_type):
        super().__init__()
        self.desired_type = desired_type

    def forward(self, data):
        assert data.dtype == self.desired_type


@pytest.mark.parametrize(
    "caster, inputs",
    itertools.product(
        [
            tt.cast.BFloat16() ** Check(torch.bfloat16),
            tt.cast.Bool() ** Check(torch.bool),
            tt.cast.Byte() ** Check(torch.uint8),
            tt.cast.Char() ** Check(torch.int8),
            tt.cast.Double() ** Check(torch.double),
            tt.cast.Float() ** Check(torch.float),
            tt.cast.Half() ** Check(torch.half),
            tt.cast.Int() ** Check(torch.int),
            tt.cast.Long() ** Check(torch.long),
            tt.cast.Short() ** Check(torch.short),
        ],
        [torch.randn(2, 3, 4)],
    ),
)
def test_cast(caster, inputs):
    caster(inputs)
