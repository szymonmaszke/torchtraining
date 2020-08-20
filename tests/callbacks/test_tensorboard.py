import os
import pathlib
import shutil
import tempfile

import torch
from torch.utils.tensorboard import SummaryWriter

import pytest
import torchtrain as tt


@pytest.mark.parametrize(
    "klass,inputs",
    [
        (tt.callbacks.tensorboard.Scalar, 15),
        (tt.callbacks.tensorboard.Scalar, torch.tensor([15]),),
        (tt.callbacks.tensorboard.Scalars, {"foo": 5, "bar": 10}),
        (
            tt.callbacks.tensorboard.Scalars,
            {"foo": torch.tensor([5]), "bar": torch.tensor(10)},
        ),
        (tt.callbacks.tensorboard.Histogram, torch.randint(0, 15, size=(10, 30))),
        (tt.callbacks.tensorboard.Image, torch.rand(3, 32, 32)),
        (tt.callbacks.tensorboard.Images, torch.rand(8, 3, 32, 32)),
        (tt.callbacks.tensorboard.Video, torch.rand(8, 6, 3, 32, 32)),
        (tt.callbacks.tensorboard.Audio, torch.rand(1, 100)),
        (tt.callbacks.tensorboard.Text, "example_text"),
    ],
)
def test_tensorboard(klass, inputs):
    directory = pathlib.Path(tempfile.mkdtemp())
    writer = SummaryWriter(directory)
    tb_writer = klass(writer, "example")
    tb_writer(inputs)
    shutil.rmtree(directory)
