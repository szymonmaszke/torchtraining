import typing

import torchtraining as tt


def get_data(batch_size: int) -> typing.Dict:
    train, validation = tt.functional.data.random_split(
        torchvision.datasets.CIFAR100(
            ".", train=True, transforms=torchvision.transforms.ToTensor()
        ),
        0.8,
        0.2,
    )
    test = torchvision.datasets.CIFAR100(
        ".", train=False, transforms=torchvision.transforms.ToTensor()
    )
    return {
        "train": torch.utils.data.DataLoader(train, batch_size, shuffle=True),
        "validation": torch.utils.data.DataLoader(validation, batch_size),
        "test": torch.utils.data.DataLoader(test, batch_size),
    }
