import torchvision
import itertools

import torch

import torchtraining as tt


def test_random_split():
    dataset = torchvision.datasets.FakeData()
    train, validation, test = tt.functional.data.random_split(dataset, 0.8, 0.1, 0.1)

    assert len(train) == 800
    assert len(validation) == 100
    assert len(test) == 100
