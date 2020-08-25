import itertools
import operator
import os
import pathlib
import shutil
import tempfile

import pytest
import torch
import torchfunc
import torchtraining as tt


@pytest.mark.parametrize(
    "iterations,comparator", itertools.product((1, 5), (operator.ge, operator.le))
)
def test_save(iterations, comparator):
    if comparator == operator.ge:
        files = iterations
    else:
        files = 1

    directory = pathlib.Path(tempfile.mkdtemp())
    module = torch.nn.Linear(20, 1)
    save = tt.callbacks.Save(module, directory / "module.py", comparator=comparator)

    for value in range(iterations):
        save.path = directory / "{}.pt".format(value)
        save(value)

    file_count = sum(len(files) for _, _, files in os.walk(directory))
    assert file_count == files
    shutil.rmtree(directory)


def test_time_stopping():
    with pytest.raises(tt.exceptions.TimeStopping) as e_info:
        stopper = tt.callbacks.TimeStopping(1)
        while True:
            stopper(1)


@pytest.mark.parametrize("iterations,nan", itertools.product((1, 5), (True, False)))
def test_terminate_nan(iterations, nan):
    if nan:
        inputs = torch.tensor([1, float("nan"), 2])
        expected = 0
    else:
        inputs = torch.ones(10)
        expected = iterations

    value = 0
    try:
        terminator = tt.callbacks.TerminateOnNan()
        for _ in range(iterations):
            terminator(inputs)
            value += 1
    except:
        pass

    assert expected == value


@pytest.mark.parametrize(
    "n,iterations,freeze", itertools.product((0, 10), (5, 10), (True, False)),
)
def test_unfreeze(n, iterations, freeze):
    expected = n < iterations
    module = torch.nn.Linear(20, 10)
    unfreezer = tt.callbacks.Unfreeze(module, n)
    if freeze:
        torchfunc.module.freeze(module)

    for i in range(iterations):
        unfreezer(i)

    for param in module.parameters():
        assert param.requires_grad == expected or not freeze
