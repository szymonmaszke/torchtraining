import torch
import torchtraining as tt


class Step(tt.steps.Step):
    def forward(self, module, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        predictions = module(images)
        loss = self.criterion(predictions, labels)

        return loss, predictions, labels


def train(optimizer, criterion, device):
    return (
        Step(criterion, gradient=True, device=device)
        ** tt.Select(loss=0)
        ** tt.pytorch.ZeroGrad(optimizer)
        ** tt.pytorch.Backward()
        ** tt.pytorch.Optimize(optimizer)
    )


def eval(criterion, device):
    return Step(criterion, gradient=False, device=device)
