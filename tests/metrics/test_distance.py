import itertools

import numpy as np
import torch

import pytest
import torchtraining as tt


@pytest.mark.parametrize(
    "metric,data",
    list(
        itertools.product(
            (tt.metrics.distance.Cosine(), tt.metrics.distance.Pairwise()),
            list(
                itertools.permutations(
                    (torch.randn(4, 10), torch.ones(4, 10), torch.zeros(4, 10),), r=2,
                )
            ),
        ),
    ),
)
def test_smoke(metric, data):
    metric(data)
