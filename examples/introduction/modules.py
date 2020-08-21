import torch


class Classifier(torch.nn.Module):
    def __init__(self, in_channels: int, labels: int):
        self.convolution = torch.nn.Sequential(
            # [batch 3, 32, 32]
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            # [batch 64, 8, 8]
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            # [batch 16, 16, 16]
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            # [batch 4, 32, 32]
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
        )
        self.linear = torch.nn.Linear(512, labels)

    def forward(self, images):
        x = self.convolution(images)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(
            images.shape[0], -1
        )
        predictions = self.linear(pooled)
        return predictions
