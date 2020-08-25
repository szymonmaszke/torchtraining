import typing

import torch
import torchtraining as tt
import torchvision


def get(batch_size: int) -> typing.Dict:
    train, validation = tt.functional.data.random_split(
        torchvision.datasets.FakeData(
            size=64,
            image_size=(3, 28, 28),
            transform=torchvision.transforms.ToTensor(),
        ),
        0.8,
        0.2,
    )
    test = torchvision.datasets.FakeData(
        size=32, image_size=(3, 28, 28), transform=torchvision.transforms.ToTensor()
    )
    return {
        "train": torch.utils.data.DataLoader(train, batch_size, shuffle=True),
        "validation": torch.utils.data.DataLoader(validation, batch_size),
        "test": torch.utils.data.DataLoader(test, batch_size),
    }
