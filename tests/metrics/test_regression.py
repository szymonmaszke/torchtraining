import itertools

import numpy as np
import torch
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_squared_error, mean_squared_log_error,
                             r2_score)

import pytest
import torchtraining as tt

# @pytest.mark.parametrize(
#     "metrics,data",
#     list(
#         itertools.product(
#             (
#                 (tt.metrics.regression.AbsoluteError(), mean_absolute_error),
#                 (tt.metrics.regression.SquaredError(), mean_squared_error),
#                 (tt.metrics.regression.SquaredLogError(), mean_squared_log_error),
#                 (tt.metrics.regression.R2(), r2_score),
#                 (tt.metrics.regression.MaxError(), max_error),
#             ),
#             list(
#                 itertools.permutations(
#                     (
#                         torch.randn(10),
#                         torch.ones(10),
#                         torch.zeros(10),
#                         torch.randn(10),
#                     ),
#                     r=2,
#                 )
#             ),
#         ),
#     ),
# )
# def test_regression(metrics, data):
#     package, scikit = metrics
#     outputs, targets = data
#     our = package(data).numpy()
#     their = scikit(targets.numpy(), outputs.numpy())
#     assert np.isclose(our, their)


@pytest.mark.parametrize(
    "metric,data",
    list(
        itertools.product(
            (
                tt.metrics.regression.AbsoluteError(),
                tt.metrics.regression.SquaredError(),
                tt.metrics.regression.SquaredLogError(),
                tt.metrics.regression.R2(),
                tt.metrics.regression.MaxError(),
                tt.metrics.regression.RegressionOfSquares(),
                tt.metrics.regression.SquaresOfResiduals(),
                tt.metrics.regression.AdjustedR2(p=6),
            ),
            (
                (torch.randn(8, 1), torch.randn(8, 1)),
                (torch.randn(2, 2, 2), torch.randn(2, 2, 2)),
            ),
        )
    ),
)
def test_smoke(metric, data):
    metric(data)


@pytest.mark.parametrize(
    "metric,data",
    [
        (tt.metrics.regression.TotalOfSquares(), torch.randn(4, 2, 3),),
        (tt.metrics.regression.TotalOfSquares(), torch.randn(4),),
    ],
)
def test_specific_smoke(metric, data):
    metric(data)
