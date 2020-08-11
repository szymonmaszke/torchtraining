import abc
import collections
import dataclasses
import typing

import torch

from . import _base


def _docstring(header, body, add_gradient: bool = False):
    body = r"""{}.

    {}

    Parameters
    ----------
    criterion : typing.Callable
        Criterion to use to get loss value. Available in `forward` as `self.criterion`
        attribute.
    """.format(
        header, body
    )

    if add_gradient:
        body += r"""
        gradient : bool
            Whether to turn gradient on/off (for training/evaluation respectively).
        """

    return (
        body
        + r"""
    device : torch.device
        Device to which tensors could be casted. Available in `forward` as
        `self.device`
    """
    )


class Step(_base.Producer):
    __doc__ = _docstring(
        "General `step`, usable both in training & evaluation.",
        "User should override `forward` method.",
        add_gradient=True,
    )

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


# Single pass through data sample
class Train(Step):
    __doc__ = _docstring(
        header="Perform user specified training step with enabled gradient.",
        body="Users should override forward method.",
    )

    def __init__(
        self, criterion: typing.Callable, device=None,
    ):
        super().__init__(criterion, True, device)


class Eval(Step):
    __doc__ = _docstring(
        header="Perform user specified evaluation step with disabled gradient.",
        body="Users should override forward method.",
    )

    def __init__(
        self, criterion: typing.Callable, device=None,
    ):
        super().__init__(criterion, False, device)
