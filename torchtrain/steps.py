import abc
import collections
import dataclasses
import typing

import torch

from . import _base, utils


@utils.steps.docstring(
    header="General `step`, usable both in training & evaluation.",
    body="User should override `forward` method.",
)
class Step(_base.Producer):
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


@utils.steps.docstring(
    header="Perform user specified training step with enabled gradient.",
    body="Users should override forward method.",
)
class Train(Step):
    def __init__(
        self, criterion: typing.Callable, device=None,
    ):
        super().__init__(criterion, True, device)


@utils.steps.docstring(
    header="Perform user specified evaluation step with disabled gradient.",
    body="Users should override forward method.",
)
class Eval(Step):
    def __init__(
        self, criterion: typing.Callable, device=None,
    ):
        super().__init__(criterion, False, device)
