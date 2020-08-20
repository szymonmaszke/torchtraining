import itertools

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

import pytest
import torchtrain as tt


@pytest.mark.parametrize(
    "functions,data",
    list(
        itertools.product(
            (
                (tt.metrics.distance.Cosine(), cosine_distances),
                (tt.metrics.distance.Euclidean(), euclidean_distances),
            ),
            list(
                itertools.permutations(
                    (torch.randn(2, 10), torch.randn(1, 10), torch.zeros(4, 10),), r=2,
                )
            ),
        ),
    ),
)
def test_distances(functions, data):
    package, scikit = functions
    outputs, targets = data
    our = package(data).numpy()
    their = scikit(targets.numpy(), outputs.numpy())
    assert np.isclose(our, their)
