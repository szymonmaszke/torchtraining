"""Perform single step on data and via specific module(s).

.. note::

    **IMPORTANT**: This module is one of core features
    so be sure to understand how it works.
    It is the core and defines how you perform single
    step through the data.

See `Introduction tutorial <https://colab.research.google.com/drive/19oI8RlpDT9JZnkW8BbFzrLL1Wse6wD5G?usp=sharing>`_ for example of `step`.

Usually it looks something along those lines::

    class Step(tt.steps.Step):
        def forward(self, module, batch):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            predictions = module(images)
            loss = self.criterion(predictions, labels)

            return loss, predictions, labels

    step = Step(criterion=torch.nn.BCEWithLogitsLoss, device=torch.device("cuda"))


.. note::

    **IMPORTANT**: You can override `__init__` if you wish to pass
    other arguments.


.. note::

    **IMPORTANT**: You can override `forward` signature to anything you
    desire. Just be sure to pass appropriate data to it (via `iteration` or `epoch`)
    or simple `__call__`.

.. note::

    **IMPORTANT**: `module` is passed from other objects and can be anything.
    In case of GANs in tutorial this is a `Tuple` of `torch.nn.Module`.


"""

import abc
import collections
import dataclasses
import typing

import torch

from . import _base, utils
from .utils import steps as steps_utils


@steps_utils.docstring(
    header="General `step`, usable both in training & evaluation.",
    body="User should override `forward` method.",
)
class Step(_base.Step):
    def __init__(
        self, criterion: typing.Callable, gradient, device=None,
    ):
        super().__init__()

        # Criterion
        self.criterion = criterion
        self.gradient = gradient
        self.device = device

    def __call__(self, *args, **kwargs):
        with torch.set_grad_enabled(self.gradient):
            return super().__call__(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


@steps_utils.docstring(
    header="Perform user specified training step with enabled gradient.",
    body="Users should override forward method.",
)
class Train(Step):
    def __init__(
        self, criterion: typing.Callable, device=None,
    ):
        super().__init__(criterion, True, device)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


@steps_utils.docstring(
    header="Perform user specified evaluation step with disabled gradient.",
    body="Users should override forward method.",
)
class Eval(Step):
    def __init__(
        self, criterion: typing.Callable, device=None,
    ):
        super().__init__(criterion, False, device)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
