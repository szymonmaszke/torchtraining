import torch

import pytest
import torchtrain as tt


@pytest.mark.parametrize(
    "loss,outputs",
    itertools.product(
        (
            tt.loss.BinaryFocal(gamma=0.2,),
            tt.loss.BinaryFocal(
                gamma=0.5, weight=torch.tensor([1.0]), pos_weight=2, reduction=torch.sum
            ),
            tt.loss.SmoothBinaryCrossEntropy(alpha=0.1),
            tt.loss.SmoothBinaryCrossEntropy(alpha=0.0),
            tt.loss.SmoothBinaryCrossEntropy(
                alpha=0.4, weight=torch.tensor([1.5], pos_weight=2, reduction=torch.sum)
            ),
        ),
        (torch.randn(4, 3, 2, 2), torch.randn(4, 8), torch.randn(1)),
    ),
)
def test_binary_smoke(loss, outputs):
    targets = torch.randn_like(outputs) < 0.5
    loss(output, targets)


@pytest.mark.parametrize(
    "loss,outputs",
    itertools.product(
        (
            tt.loss.MulticlassFocal(gamma=0.2,),
            tt.loss.MulticlassFocal(
                gamma=0.5,
                weight=torch.tensor([1.0]),
                ignore_index=1,
                reduction=torch.sum,
            ),
            tt.loss.SmoothCrossEntropy(alpha=0.1),
            tt.loss.SmoothCrossEntropy(alpha=0.0),
            tt.loss.SmoothCrossEntropy(
                alpha=0.4,
                weight=torch.tensor([1.5], ignore_index=0, reduction=torch.sum),
            ),
        ),
        (torch.randn(4, 3, 7), torch.randn(4, 8)),
    ),
)
def test_binary_smoke(loss, outputs):
    targets = torch.argmax(outputs, dim=-1)
    loss(output, targets)
