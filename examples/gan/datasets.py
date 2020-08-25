import random

import torch


class Noise(torch.utils.data.Dataset):
    def __init__(self, batch_size: int, features: int):
        self.batch_size = batch_size
        self.features = features

    def __getitem__(self, _):
        return torch.randn(self.batch_size, self.features), torch.zeros(self.batch_size)


# Add noise?
class CIFAR10AndFakeImages(torch.utils.data.Dataset):
    def __init__(self, cifar, p: float = 0.5):
        self.cifar = cifar
        self.p = p
        # Here generated fake_images will be added
        self.fake_images = None

    # Last generated images will be saved here
    def add_fake_images(self, fake_images):
        self.fake_images = fake_images

    def __getitem__(self, index):
        if random.random() < self.p:
            if not self.fake_images[index % self.fake_images.shape[0]]:
                raise ValueError(
                    "Got no fake generated images, please run generator first."
                )
            return self.fake_images[index % self.fake_images.shape[0]]
        return self.cifar[index]
