import abc
import typing

import torch

from ._base import Accumulator


class Sum(Accumulator):
    """Sum data coming into this object.

    Accumulated data will be returned at each step.

    `data` should have `+=` operator implemented between it's instances
    and Python integers.

    Arguments
    ---------
    data: Any
        Anything which has `__iadd__`/`__add__` operator implemented between it's instances
        and Python integers.

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
        """Assign `0` to `self.data` clearing `saver`"""
        self.data = 0

    def forward(self, data) -> None:
        self.data += data

    def calculate(self) -> typing.Any:
        return self.data


class Mean(Accumulator):
    """Sum data coming into this object.

    Accumulated data will be returned at each step.

    `data` should have `+=` operator implemented between it's instances
    and Python integers.

    Arguments
    ---------
    data: Any
        Anything which has `__iadd__`/`__add__` operator implemented between it's instances
        and Python integers. It should also have `__div__` operator implemented
        for proper mean calculation.

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
        self._counter += 1
        self.data += data

    def calculate(self) -> typing.Any:
        return self.data / self._counter


class List(Accumulator):
    """Sum data coming into this object.

    **It is advised NOT TO USE this accumulator due to memory inefficiencies!**

    List containing data received up to this moment will be returned
    at every `step`.

    `data` **does not** have to implement any concept
    (as it is only appended to `list`).

    Arguments
    ---------
    data: Any
        Anything which can be added to `list`. So anything I guess

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
        return self.data

    def accumulate(self, data) -> None:
        self.data.append()


class Except(Accumulator):
    """Special modifier of accumulators accumulating every value except specified.

    **One of `begin`, `end` has to be specified!**


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

    Arguments
    ---------
    data: Any
        Anything which `accumulator` can consume

    Returns
    -------
    Any
        Whatever `accumulator` returns after accumulation. At each step proper
        value up to this point is returned nonetheless. Usually `torch.Tensor`
        or `list`.

    """

    def __init__(
        self,
        accumulator: Accumulator,
        begin: typing.Union[int, torch.Tensor] = None,
        end: typing.Union[int, torch.Tensor] = None,
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
        self._counter = -1
        self.accumulator.reset()

    def forward(self, data) -> None:
        self._counter += 1
        if not self._in(self._counter):
            self.accumulator(data)

    def calculate(self) -> typing.Any:
        return self.accumulator.calculate()
