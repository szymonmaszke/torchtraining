import typing

import torch
import yaml

import loguru

from . import (accumulators, callbacks, cast, device, epochs, exceptions,
               functional, iterations, loss, metrics, pytorch, quantization,
               steps)
from ._base import Accumulator, GeneratorProducer, Operation, Producer
from ._version import __version__

loguru.logger.level("NONE", no=0)


class Select(Operation):
    """Select output item(s) returned from `step` or `iteration` objects.

    Allows users to focus on specific part of output and pipe this specific
    values to other operations (like metrics, loggers etc.).

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Generate loss and other necessary items
                ...
                return loss, predictions, targets


        step = TrainStep(criterion, device)
        # Select `loss` and perform backpropagation
        step > tt.Select(loss=0) > tt.pytorch.Backward()


    Parameters
    ----------
    output_selection : **output_selection
        `name`: output_index mapping selecting which element from step returned `tuple`
        to choose. `name` can be arbitrary, but should be named like the variable
        returned from `step`. See example above.

    Arguments
    ---------
    data: Iterable[Any]
        Iterable (e.g. `tuple`, `list`) with any elements.

    Returns
    -------
    Any | List[Any]
        If single int is passed `output_selection` return single element from `Iterable`.
        Otherwise returns chosen elements as `list`

    """

    def __init__(self, **output_selection: int):
        if len(output_selection) > 0:
            raise ValueError(
                "{}: At least one output index has to be specified, got {} output indices.".format(
                    self, len(output_selection)
                )
            )
        self.output_selection = output_selection

        if len(self.output_selection) == 1:
            self._selection_method = lambda data: data[
                self.output_selection.values()[0]
            ]
        else:
            self._selection_method = lambda data: [
                data[index] for index in self.output_selection.values()
            ]

    def forward(self, data: typing.Iterable[typing.Any]) -> typing.Any:
        return self._selection_method(data)

    def __str__(self) -> str:
        return yaml.dump({super().__str__(): self.output_selection})


class Split(Operation):
    """Split pipe with data to multiple components.

    Useful when users wish to use results in multiple places.
    Example calculating metrics and logging them::


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Generate loss and other necessary items
                ...
                # Assume binary classification
                return loss, logits, targets


        step = TrainStep(criterion, device)

        # Push (logits, targets) to Precision and Recall
        # and log those values
        step > tt.Select(logits=1, targets=2) > tt.Split(
            tt.metrics.classification.binary.Precision() > tt.callbacks.Logger("Precision"),
            tt.metrics.classification.binary.Recall() > tt.callbacks.Logger("Recall"),
        )


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


class Flatten(Operation):
    r"""Flatten arbitrarily nested data.

    Single `tuple` with all elements (not being `tuple` or `list`).

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                module1, module2 = module
                ...
                return ((logits1, targets1), (logits2, targets2), module1, module2)


        step = TrainStep(criterion, device)

        # Tuple (logits1, targets1, logits2, targets2, module1, module2)
        step > tt.Flatten()

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


class If(Operation):
    """Run operation only If `condition` is `True`.

    `condition` can also be a single argument callable, in this case it can be
    dependent on data, see below::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss


        step = TrainStep(criterion, device)
        step > tt.If(lambda loss: loss > 10, tt.callbacks.Logger("VERY HIGH LOSS!!!"))

    Parameters
    ----------
    condition: bool | Callable(Any) -> bool
        If boolean value and if `true`, run underlying Op (or other Callable).
        If Callable, should take data as argument and return decision based on
        that as single `bool`.
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

    def __init__(
        self,
        condition: typing.Union[bool, typing.Callable[[typing.Any], bool,]],
        op: typing.Callable[[typing.Any,], typing.Any],
    ):
        if isinstance(condition, bool):
            self._choice_method = lambda data: condition
        else:
            self._choice_method = lambda data: condition(data)

        self.condition = condition
        self.op = op

    def forward(self, data: typing.Any) -> typing.Any:
        if self.condition(data):
            return self.op(data)
        return data

    def __str__(self) -> str:
        if self.condition:
            return str(self.op)
        return "no-op"


# Make it dynamic and static
class IfElse(Operation):
    """Run `operation1` only if `condition` is `True`, otherwise run `operation2`.

    `condition` can also be a single argument callable, in this case it can be
    dependent on data, see below::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss


        step = TrainStep(criterion, device)

        step > tt.If(
            lambda loss: loss > 10,
            tt.callbacks.Logger("VERY HIGH LOSS!!!"),
            tt.callbacks.Logger("LOSS IS NOT THAT HIGH..."),
        )

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


class ToAll(Operation):
    r"""Apply pipe to each element of sample.**

    **Important:**

    If you want to apply pipe to all nested elements (e.g. in nested `tuple`),
    please use `torchtrain.Flatten` object first.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss


        step = TrainStep(criterion, device)

        step > tt.If(
            lambda loss: loss > 10,
            tt.callbacks.Logger("VERY HIGH LOSS!!!"),
            tt.callbacks.Logger("LOSS IS NOT THAT HIGH..."),
        )


    Parameters
    ----------
    pipe : Callable
        Pipe to apply to each element of sample.

    Arguments
    ---------
    data: Any
        Anything you want (usually `torch.Tensor` like stuff).

    Returns
    -------
    Tuple[pipe(subsample)]
        Tuple consisting of subsamples with pipe applied.

    """

    def __init__(self, pipe: typing.Callable):
        self.pipe = pipe

    def forward(self, sample):
        return tuple(self.pipe(subsample) for subsample in sample)


class Lambda(Operation):
    """Run user specified pipe on `data`.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return accuracy


        step = TrainStep(criterion, device)

        # If you want to get that SOTA badly, we got ya covered
        step > tt.Lambda(lambda accuracy: accuracy * 2)

    Parameters
    ----------
    pipe : Callable(Any) -> Any
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
        Value returned from `pipe`

    """

    def __init__(
        self,
        pipe: typing.Callable[[typing.Any,], typing.Any],
        name: str = "torchtrain.Lambda",
    ):
        self.pipe = pipe
        self.name = name

    def __str__(self) -> str:
        return self.name

    def forward(self, data: typing.Any) -> typing.Any:
        return self.pipe(data)
