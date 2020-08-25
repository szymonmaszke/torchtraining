"""Accumulate results from `iterations` or `epochs`

.. note::

    **IMPORTANT**: This module is one of core features
    so be sure to understand how it works.

.. note::

    **IMPORTANT**: Accumulators should be applied to `iteration`
    objects. This way those can efficiently accumulate value later passed
    to other operations.

Example::

    iteration
    ** tt.Select(predictions=1, labels=2)
    ** tt.metrics.classification.multiclass.Accuracy()
    ** tt.accumulators.Mean()
    ** tt.Split(
        tt.callbacks.Log(f"{name} Accuracy"),
        tt.callbacks.tensorboard.Scalar(writer, f"{name}/Accuracy"),
    )

Code above will accumulate `accuracy` from each step and after `iteration`
ends it will be send to `tt.Split`.


.. note::

    **IMPORTANT**: If users wish to implement their own `accumulators`
    `forward` shouldn't return anything but accumulate data in `self.data`
    variable. No argument `calculate` should return `self.data` after
    calculating accumulated value (e.g. for `mean` it would be division
    by number of samples).

"""

import abc
import typing

import torch

from ._base import Accumulator


class Sum(Accumulator):
    """Sum data coming into this object.

    `data` should have `+=` operator implemented between it's instances
    and Python integers.

    .. note::

        **IMPORTANT**: This is one of memory efficient accumulators
        and can be safely used.

    Returns
    -------
    torch.Tensor | Any
        Sum of values after accumulation. At each step proper
        summation up to this point is returned nonetheless.
        `torch.Tensor` usually, but can be anything "summable".

    """

    def __init__(self):
        super().__init__()
        self.data = 0

    def reset(self) -> None:
        """Assign 0 to `self.data` clearing `saver`."""
        self.data = 0

    def forward(self, data) -> None:
        """
        Arguments
        ---------
        data: Any
            Anything which has `__iadd__`/`__add__` operator implemented between it's instances
            and Python integers.
        """
        self.data += data

    def calculate(self) -> typing.Any:
        """Calculate final value.

        Returns
        -------
        torch.Tensor
            Data accumulated via addition.

        """
        return self.data


class Mean(Accumulator):
    """Take mean of the data coming into this object.

    `data` should have `+=` operator implemented between it's instances
    and Python integers.

    .. note::

        **IMPORTANT**: This is one of memory efficient accumulators
        and can be safely used. Should be preferred over accumulating
        data via `torchtraining.accumulators.List`


    Returns
    -------
    torch.Tensor | Any
        Mean of values after accumulation. At each step proper
        mean up to this point is returned nonetheless.
        `torch.Tensor` usually, but can be anything implementing concept above.

    """

    def __init__(self):
        super().__init__()
        self.data = 0
        self._counter = 0

    def reset(self) -> None:
        """Assign `0` to `self.data` and zero out counter clearing `saver`"""
        self.data = 0
        self._counter = 0

    def forward(self, data: typing.Any) -> None:
        """
        Arguments
        ---------
        data: Any
            Anything which has `__iadd__`/`__add__` operator implemented between it's instances
            and Python integers. It should also have `__div__` operator implemented
            for proper mean calculation.
        """
        self._counter += 1
        self.data += data

    def calculate(self) -> typing.Any:
        """Calculate final value.

        Returns
        -------
        torch.Tensor
            Accumulated data after summation and division by number of samples.

        """
        return self.data / self._counter


class List(Accumulator):
    """Sum data coming into this object.

    .. note::

        **IMPORTANT**: It is advised **NOT TO USE** this accumulator
        due to memory inefficiencies. Prefer `torchtraining.accumulators.Sum`
        or `torchtraining.accumulators.Mean` instead.

    List containing data received up to this moment.
    `data` **does not** have to implement any concept
    (as it is only appended to `list`).


    Returns
    -------
    List
        List of values after accumulation. At each step proper
        `list` up to this point is returned nonetheless.

    """

    def __init__(self):
        super().__init__()
        self.data = []

    def reset(self) -> None:
        """Assign empty `list` to `self.data clearing `saver`"""
        self.data = []

    def forward(self) -> typing.List[typing.Any]:
        """
        Arguments
        ---------
        data: Any
            Anything which can be added to `list`. So anything I guess
        """
        return self.data

    def accumulate(self, data) -> None:
        """Calculate final value.

        Returns
        -------
        torch.Tensor
            Return `List` with gathered data.

        """
        self.data.append()


class Except(Accumulator):
    """Special modifier of accumulators accumulating every value except specified.

    .. note::

        **IMPORTANT**: One of the `begin`, `end` has to be specified.

    .. note::

        **IMPORTANT**: This accumulators is useful in conjunction
        with `torchtraining.iterations.Multi` (e.g. for GANs and other irregular
        type of training).

    User can effectively choose which data coming from step should be accumulated
    and can divide accumulation based on that.

    Parameters
    ----------
    accumulator: tt.Accumulator
        Instance of accumulator to use for `data` accumulation.
    begin: int | torch.Tensor[int], optional
        If `int`, it should specify beginning of incoming values stream which
        will not be taken into accumulation. If `torch.Tensor` containing integers,
        it should specify consecutive beginnings of streams which are not taken into account.
        If left unspecified (`None`), `begin` is assumed to be `0`th step.
        Every modulo element of stream matching [begin, end] range will not be
        forwarded to accumulator.
    end: int | torch.Tensor[int], optional
        If `int`, it should specify end of incoming values stream which
        will not be taken into accumulation. If `torch.Tensor` containing integers,
        it should specify consecutive ends of stream which will not be taken into account.
        If left unspecified (`None`), `end` is assumed to be the same as `begin`.
        This effectively excludes every `begin` element coming from value stream.
        Every modulo element of stream matching [begin, end] range will not be
        forwarded to accumulator.


    Returns
    -------
    Any
        Whatever `accumulator` returns after accumulation. At each step proper
        value up to this point is returned nonetheless. Usually `torch.Tensor`
        or `list`.

    """

    def __init__(
        self, accumulator: Accumulator, begin=None, end=None,
    ):
        def _validate_argument(arg, name: str):
            if torch.is_tensor(arg):
                if not arg.dtype in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError(
                        "{} as torch.Tensor should be of integer-like type, got: {}".format(
                            name, arg.dtype
                        )
                    )
            if not isinstance(arg, int):
                raise ValueError(
                    "{} should be torch.Tensor/int, got : {}".format(name, type(arg))
                )
            return arg

        def _same_type(arg1, arg2):
            if not isinstance(arg1, type(arg2)):
                raise ValueError(
                    "begin and end should be of the same type. Got {} for begin and {} for end".format(
                        type(arg1), type(arg2)
                    )
                )

        def _same_shape(arg1, arg2):
            if arg1.shape != arg2.shape:
                raise ValueError(
                    "begin and end should have the same shape. Got {} for begin and {} for end".format(
                        arg1.shape, arg2.shape
                    )
                )

        def _one_of_specified(arg1, arg2):
            if arg1 is None and arg2 is None:
                raise ValueError("One of begin/end has to be specified.")

        super().__init__()

        _one_of_specified(begin, end)
        self.begin = _validate_argument(begin, "begin")
        if end is None:
            self.end = self.begin
        else:
            _same_type(self.begin, end)
            self.end = _validate_argument(end, "end")
            _same_shape(begin, end)

        self.accumulator = accumulator
        if isinstance(self.begin, int):
            self._in = lambda i: (self.begin <= (i % self.end + 1) <= self.end) and (
                i != self.end + 1
            )
        else:
            maximum = torch.max(self.end)
            self._in = lambda i: (
                (self.begin <= (i % maximum + 1)) & ((i % maximum + 1) <= self.end)
            ).any() & (i != maximum + 1)

        self._counter = -1

    def reset(self) -> None:
        """Reset internal `accumulator`."""
        self._counter = -1
        self.accumulator.reset()

    def forward(self, data) -> None:
        """
        Arguments
        ---------
        data: Any
            Anything which `accumulator` can consume
        """
        self._counter += 1
        if not self._in(self._counter):
            self.accumulator(data)

    def calculate(self) -> typing.Any:
        """Calculate final value.

        Returns
        -------
        Any
            Returns anything `accumulator` accumulated.

        """
        return self.accumulator.calculate()
