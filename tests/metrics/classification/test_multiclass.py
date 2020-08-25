import itertools

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             fbeta_score, jaccard_score, matthews_corrcoef,
                             precision_score, recall_score)

import pytest
import torchtraining as tt
import torchtraining.metrics.classification.multiclass as M

# @pytest.mark.parametrize(
#     "metrics,data",
#     list(
#         itertools.product(
#             (
#                 (M.Accuracy(), accuracy_score),
#                 (M.Jaccard(), jaccard_score),
#                 (M.Recall(), recall_score),
#                 (M.Precision(), precision_score),
#                 (M.BalancedAccuracy(), balanced_accuracy_score),
#                 (M.F1(), f1_score),
#                 (
#                     M.FBeta(beta=4),
#                     lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=4),
#                 ),
#                 (M.MatthewsCorrelationCoefficient(), matthews_corrcoef),
#             ),
#             (torch.randn(10, 5), torch.randint(high=2, size=(10,))),
#         ),
#     ),
# )
# def test_multiclass(metrics, data):
#     package, scikit = metrics
#     outputs, targets = data
#     our = package(data).numpy()
#     their = scikit(targets.numpy(), torch.argmax(outputs, dim=-1).numpy())
#     assert np.isclose(our, their)


@pytest.mark.parametrize(
    "metric,data",
    list(
        itertools.product(
            (
                M.Accuracy(),
                M.Jaccard(),
                M.TruePositive(),
                M.FalsePositive(),
                M.FalsePositive(),
                M.TrueNegative(),
                M.FalseNegative(),
                # Special handling
                M.ConfusionMatrix(),
                M.Recall(),
                M.Specificity(),
                M.Precision(),
                M.NegativePredictiveValue(),
                M.FalseNegativeRate(),
                M.FalsePositiveRate(),
                M.FalseDiscoveryRate(),
                M.FalseOmissionRate(),
                M.CriticalSuccessIndex(),
                M.BalancedAccuracy(),
                M.F1(),
                M.FBeta(beta=2.4),
                M.MatthewsCorrelationCoefficient(),
            ),
            (
                (torch.randn(4, 10), torch.randint(high=2, size=(4,))),
                (torch.randn(4, 4, 5), torch.randint(high=2, size=(4, 5))),
            ),
        )
    ),
)
def test_smoke(metric, data):
    metric(data)
