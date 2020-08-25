import pytest
import torch
import torchtraining as tt


class Step(tt.steps.Train):
    def forward(self, module, sample):
        data, target = sample
        predictions = module(data).squeeze()
        loss = self.criterion(predictions, target)
        return loss, predictions, target


def test_step():
    module = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(module.parameters())
    step = (
        Step(torch.nn.MSELoss())
        ** tt.Select(loss=0)
        ** tt.pytorch.ZeroGrad(optimizer)
        ** tt.pytorch.Backward(accumulate=2)
        ** tt.pytorch.Optimize(optimizer, accumulate=2)
        ** tt.pytorch.Detach()
        ** tt.callbacks.Log("Loss")
    )

    step = Step(torch.nn.MSELoss()) ** tt.Select(loss=0)

    step ** tt.Select(
        predictions=1, target=2
    ) ** tt.metrics.regression.MaxError() ** tt.callbacks.Log(name="Max Error")

    for _ in range(5):
        step(module, (torch.randn(8, 10), torch.randn(8)))
