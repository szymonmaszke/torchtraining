import typing

import torch
import yaml

import loguru

from . import (callbacks, cast, device, epochs, functional, iterations, loss,
               metrics, operations, ops, quantization, savers, steps)
from ._base import GeneratorProducer, Op, Producer, Saver
from ._version import __version__

loguru.logger.level("NONE", no=0)


class Select(Op):
    """Select output item returned from `step` or `iteration` objects.

    Allows users to focus on specific output from result generator and
    post-process it (e.g. piping part of output to metric, loggers or a-like).

    Parameters
    ----------
    output_indices : *int
        Zero-based index into output tuple choosing part of output to propagate
        further down the pipeline.
        At least one index has to be specified.

    Arguments
    ---------
    data: Iterable[Any]
        Iterable (e.g. `tuple`, `list`) with any elements.

    Returns
    -------
    Any | List[Any]
        If single int is passed `output_indices` return single element from `Iterable`.
        Otherwise returns chosen elements as `list`

    """

    def __init__(self, *output_indices: int):
        if len(output_indices) > 0:
            raise ValueError(
                "{}: At least one output index has to be specified, got {} output indices.".format(
                    self, len(output_indices)
                )
            )
        self.output_indices = output_indices

        if len(self.output_indices) == 1:
            self._selection_method = lambda data: data[self.output_indices[0]]
        else:
            self._selection_method = lambda data: [
                data[index] for index in self.output_indices
            ]

    def forward(self, data: typing.Iterable[typing.Any]) -> typing.Any:
        return self._selection_method(data)

    def __str__(self) -> str:
        return yaml.dump({super().__str__(): self.output_indices})


class Split(Op):
    """Split pipe with data to multiple components.

    Useful when users wish to log results of runner to multiple places.
    Example output logging to `tensorboard`, `stdout` and `file`::

        import torchtrain as tt

    Parameters
    ----------
    *operations: Callable(data) -> Any
        Operations to which results will be passed.
    return_modified: bool, optional
        Return outputs from `operations` as a `list` if `True`. If `False`, returns
        original `data` passed into `Split`. Default: `False`

    Arguments
    ---------
    data: Any
        Data which will be passed to provided `operations`.

    Returns
    -------
    data | List[modified data]
        Returns `data` passed originally or `list` containing modified data
        returned from `operations`.

    """

    def __init__(
        self,
        *operations: typing.Callable[[typing.Any,], typing.Any],
        return_modified: bool = False
    ):
        if not operations:
            raise ValueError("Split requires at least one operation to pass data into.")
        self.operations = operations
        self.return_modified = return_modified

    def forward(
        self, data: typing.Any
    ) -> typing.Union[typing.Any, typing.List[typing.Any]]:
        processed_data = []
        for op in self.operations:
            result = op(data)
            if self.return_modified:
                processed_data.append(result)
        if self.return_modified:
            return processed_data
        return data

    def __str__(self) -> str:
        return yaml.dump({super().__str__(): self.operations})


class Flatten(Op):
    r"""Flatten arbitrarily nested data.

    Single `tuple` with all elements (not being `tuple` or `list`).

    Parameters
    ----------
    types : Tuple[type], optional
        Types to be considered non-flat. Those will be recursively flattened.
        Default: `(list, tuple)`

    Arguments
    ---------
    data: Iterable[Iterable ... Iterable[Any]]
        Arbitrarily nested data being one of type provided in `types`.

    Returns
    -------
    Tuple[samples]
        Single `tuple` with all elements (not being `tuple` or `list`).

    """

    def __init__(self, types: typing.Tuple = (list, tuple)):
        self.types = types

    def forward(self, sample) -> typing.List[typing.Any]:
        if not isinstance(sample, self.types):
            return sample
        return Flatten._flatten(sample, self.types)

    @staticmethod
    def _flatten(
        items: typing.Iterable[typing.Any], types: typing.Tuple
    ) -> typing.List[typing.Any]:
        if isinstance(items, tuple):
            items = list(items)

        for index, x in enumerate(items):
            while index < len(items) and isinstance(items[index], types):
                items[index : index + 1] = items[index]
        return items


class If(Op):
    """Run operation only If `condition` is `True`.

    Parameters
    ----------
    condition: bool
        Boolean value. If `true`, run underlying Op (or other Callable).
    op: torchtrain.Op | Callable
        Operation or single argument callable to run in...

    Arguments
    ---------
    data: Any
        Anything you want (usually `torch.Tensor` like stuff).

    Returns
    -------
    Any
        If `true`, returns value from `op`, otherwise passes original `data`

    """

    def __init__(self, condition: bool, op: typing.Callable[[typing.Any,], typing.Any]):
        self.condition = condition
        self.op = op

    def forward(self, data: typing.Any) -> typing.Any:
        if self.condition:
            return self.op(data)
        return data

    def __str__(self) -> str:
        if self.condition:
            return str(self.op)
        return "no-op"


class IfElse(Op):
    """Run `operation1` only if `condition` is `True`, otherwise run `operation2`.

    Parameters
    ----------
    condition: bool
        Boolean value. If `true`, run underlying Op (or other Callable).
    operation1: torchtrain.Op | Callable
        Operation or callable getting single argument (`data`) and returning anything.
    operation2: torchtrain.Op | Callable
        Operation or callable getting single argument (`data`) and returning anything.

    Arguments
    ---------
    data: Any
        Anything you want (usually `torch.Tensor` like stuff).

    Returns
    -------
    Any
        If `true`, returns value from `operation1`, otherwise from `operation2`.

    """

    def __init__(
        self,
        condition: bool,
        op1: typing.Callable[[typing.Any,], typing.Any],
        op2: typing.Callable[[typing.Any,], typing.Any],
    ):
        self.condition = condition
        self.op1 = op1
        self.op2 = op2

    def forward(self, data: typing.Any) -> typing.Any:
        if self.condition:
            return self.op1(data)
        return self.op2(data)

    def __str__(self) -> str:
        if self.condition:
            return str(self.op1)
        return str(self.op2)


class Lambda(Op):
    """Run user specified function on single item.

    Parameters
    ----------
    function : Callable(Any) -> Any
        Single argument callable getting data and returning some value.
    name : str, optional
        `string` representation of this operation (if any).
        Default: `torchtrain.metrics.Lambda`

    Arguments
    ---------
    data: Any
        Anything you want (usually `torch.Tensor` like stuff).

    Returns
    -------
    Any
        Value returned from `function`

    """

    def __init__(
        self,
        function: typing.Callable[[typing.Any,], typing.Any],
        name: str = "torchtrain.Lambda",
    ):
        self.function = function
        self.name = name

    def __str__(self) -> str:
        return self.name

    def forward(self, data: typing.Any) -> typing.Any:
        return self.function(data)
