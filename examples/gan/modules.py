import torch


class Generator(torch.nn.Module):
    def __init__(self, in_channels: int):
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(512, 4096), torch.nn.BatchNorm1d(4096)
        )
        self.convolution = torch.nn.Sequential(
            # [batch 1024, 2, 2]
            torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.PixelShuffle(2),
            # [batch 256, 4, 4]
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.PixelShuffle(2),
            # [batch 64, 8, 8]
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.PixelShuffle(2),
            # [batch 16, 16, 16]
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.PixelShuffle(2),
            # [batch 4, 32, 32]
            torch.nn.Conv2d(4, in_channels, kernel_size=3, padding=1),
            # [batch 3, 32, 32] -> image
            torch.nn.Sigmoid(),
        )

    # noise shape: [batch, 512]
    def forward(self, noise):
        x = self.linear(noise)
        # [batch, 4096]
        x_image = x.reshape(-1, 1024, 2, 2)
        # [batch, 1024, 2, 2]
        image = self.convolution(x_image)
        # [batch, 3, 32, 32]
        return image


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels: int):
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
        self.linear = torch.nn.Linear(512, 1)

    def forward(self, images):
        x = self.convolution(images)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(
            images.shape[0], -1
        )
        prediction = self.linear(pooled)
        return prediction
