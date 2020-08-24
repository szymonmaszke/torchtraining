import torch


class Classifier(torch.nn.Module):
    def __init__(self, in_channels: int, labels: int):
        super().__init__()
        self.convolution = torch.nn.Sequential(
            # [batch 3, 32, 32]
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            # [batch 64, 8, 8]
        )
        self.linear = torch.nn.Linear(64, labels)

    def forward(self, images):
        x = self.convolution(images)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(
            images.shape[0], -1
        )
        predictions = self.linear(pooled)
        return predictions
