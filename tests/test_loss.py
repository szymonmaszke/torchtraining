import itertools

import torch

import pytest
import torchtraining as tt


@pytest.mark.parametrize(
    "loss,outputs",
    list(
        itertools.product(
            (
                tt.loss.BinaryFocal(gamma=0.2,),
                tt.loss.BinaryFocal(
                    gamma=0.5,
                    weight=1.2,
                    pos_weight=torch.tensor(2),
                    reduction=torch.sum,
                ),
                tt.loss.SmoothBinaryCrossEntropy(alpha=0.1),
                tt.loss.SmoothBinaryCrossEntropy(alpha=0.0),
                tt.loss.SmoothBinaryCrossEntropy(
                    alpha=0.4,
                    weight=1.5,
                    pos_weight=torch.tensor(2),
                    reduction=torch.sum,
                ),
            ),
            (torch.randn(4, 3, 2, 2), torch.randn(4, 8), torch.randn(1)),
        )
    ),
)
def test_binary_smoke(loss, outputs):
    targets = torch.randn_like(outputs) < 0.5
    loss(outputs, targets)


@pytest.mark.parametrize(
    "loss,outputs",
    list(
        itertools.product(
            (
                tt.loss.MulticlassFocal(gamma=0.2,),
                tt.loss.MulticlassFocal(
                    gamma=0.5, ignore_index=1, reduction=torch.sum,
                ),
                tt.loss.SmoothCrossEntropy(alpha=0.1),
                tt.loss.SmoothCrossEntropy(alpha=0.0),
                tt.loss.SmoothCrossEntropy(
                    alpha=0.4, ignore_index=0, reduction=torch.sum,
                ),
            ),
            (torch.randn(4, 7, 3), torch.randn(4, 7), torch.randn(4, 8, 3, 2)),
        )
    ),
)
def test_multiclass_smoke(loss, outputs):
    targets = torch.argmax(outputs, dim=1)
    loss(outputs, targets)
