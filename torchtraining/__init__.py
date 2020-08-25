"""Root module of `torchtraining` containing common operations.

.. note::

    **IMPORTANT**: This module is one of the most important and
    is used in almost any DL task so be sure to understand it!


Operations in this module can be used to:

    * control pipeline flow
    * select output from `step`s
    * send data to multiple operations

See below for more info.

"""

import typing

import loguru
import torch
import yaml

from . import (accelerators, accumulators, callbacks, cast, device, epochs,
               exceptions, functional, iterations, loss, metrics, pytorch,
               quantization, steps)
from ._base import (Accumulator, Epoch, GeneratorProducer, Iteration,
                    Operation, ReturnProducer, Step)
from ._version import __version__

loguru.logger.level("NONE", no=0)


class Select(Operation):
    """Select output item(s) returned from `step` or `iteration` objects.

    Allows users to focus on specific part of output and pipe specified
    values to other operations (like `metrics`, `loggers` etc.).

    .. note::

        **IMPORTANT**: This operation is run in almost any case
        so be sure to understand how it works.


    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Generate loss and other necessary items
                ...
                return loss, predictions, targets


        step = TrainStep(criterion, device)
        # Select `loss` and perform backpropagation
        # Only single value will be forward to backward from
        # (loss, predictions, targets) tuple
        step ** tt.Select(loss=0) ** tt.pytorch.Backward()

    .. note::

        Name of keyword argument can be arbitrary but
        **you are really encouraged** to name it like the variable
        returned from `step` (or at least make it understandable to others).


    Parameters
    ----------
    output_selection : **output_selection
        `name`: output_index mapping selecting which element from step returned `tuple`
        to choose. `name` can be arbitrary, but should be named like the variable
        returned from `step`. See example above.


    Returns
    -------
    Any | List[Any]
        If single int is passed `output_selection` return single element from `Iterable`.
        Otherwise returns chosen elements as `list`

    """

    def __init__(self, **output_selection: int):
        if len(output_selection) < 1:
            raise ValueError(
                "{}: At least one output index has to be specified, got {} output indices.".format(
                    self, len(output_selection)
                )
            )

        super().__init__()
        self.output_selection = output_selection

        if len(self.output_selection) == 1:
            self._selection_method = lambda data: data[
                list(self.output_selection.values())[0]
            ]
        else:
            self._selection_method = lambda data: [
                data[index] for index in list(self.output_selection.values())
            ]

    def forward(self, data: typing.Iterable[typing.Any]) -> typing.Any:
        """
        Arguments
        ---------
        data: Iterable[Any]
            Iterable (e.g. `tuple`, `list`) with any elements.
        """
        selected = self._selection_method(data)
        return selected

    def __str__(self) -> str:
        return yaml.dump({super().__str__(): self.output_selection})


class Split(Operation):
    """Split operation with data to multiple components.

    .. note::

        **IMPORTANT**: This operation is run in almost any case
        so be sure to understand how it works.


    Useful when users wish to use calculated result in multiple places.
    Example calculating metrics and logging them in multiple places::


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Generate loss and other necessary items
                ...
                # Assume binary classification
                return loss, logits, targets


        step = TrainStep(criterion, device)

        # Push (logits, targets) to Precision and Recall
        # and log those values after calculating metrics
        step ** tt.Select(logits=1, targets=2) ** tt.Split(
            tt.metrics.classification.binary.Precision() ** tt.callbacks.Logger("Precision"),
            tt.metrics.classification.binary.Recall() ** tt.callbacks.Logger("Recall"),
        )


    Parameters
    ----------
    *operations: Callable(data) -> Any
        Operations to which results will be passed.
    return_modified: bool, optional
        Return outputs from `operations` as a `list` if `True`. If `False`, returns
        original `data` passed into `Split`. Default: `False`

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
        super().__init__()
        self.operations = operations
        self.return_modified = return_modified

    def forward(
        self, data: typing.Any
    ) -> typing.Union[typing.Any, typing.List[typing.Any]]:
        """
        Arguments
        ---------
        data: Any
            Data which will be passed to provided `operations`.
        """
        processed_data = []
        for operation in self.operations:
            result = operation(data)
            if self.return_modified:
                processed_data.append(result)
        if self.return_modified:
            return processed_data
        return data

    def __str__(self) -> str:
        return yaml.dump({super().__str__(): self.operations})


class OnSplittedTensor(Operation):
    """Split tensor along dimension and apply operation on each element.

    By default, `torch.Tensor` will be splitted along batch (`dim=0`)

    .. note::

        **IMPORTANT**: After splitting first dimension is squeezed
        via `torch.squeeze`.


    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Dummy step
                images, labels
                return images


        step = TrainStep(criterion, device)

        # Assume summary_writer is instance of torch.utils.tensorboard.SummaryWriter()
        step ** tt.OnSplittedTensor(tt.callbacks.tensorboard.Image(summary_writer))
        # Each image will be saved separately



    Parameters
    ----------
    operation: tt.Operation | Callable(data) -> Any
        Operation which will be applied to each element of `torch.Tensor`.
    dim: int, optional
        Dimension along which `data` `torch.Tensor` will be splitted.
        Default: `0`


    Returns
    -------
    Tuple[torch.Tensor]
        Splitted `data` along `dim` (unmodified by `operation`).

    """

    def __init__(self, operation: Operation, dim: int = 0):
        super().__init__()
        self.operation = operation
        self.dim = dim

    def forward(self, data):
        """
        Arguments
        ---------
        data: torch.Tensor
            Tensor to split and apply `operation` on.
        """
        splitted = tuple(map(torch.squeeze, torch.split(data, 1, self.dim)))
        for tensor in splitted:
            self.operation(splitted)
        return splitted


class Flatten(Operation):
    r"""Flatten arbitrarily nested data.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                module1, module2 = module
                ...
                return ((logits1, targets1), (logits2, targets2), module1, module2)


        step = TrainStep(criterion, device)

        # Tuple (logits1, targets1, logits2, targets2, module1, module2)
        step ** tt.Flatten()

    Parameters
    ----------
    types : Tuple[type], optional
        Types to be considered non-flat. Those will be recursively flattened.
        Default: `(list, tuple)`

    Returns
    -------
    Tuple[samples]
        Single `tuple` with all elements (not being `tuple` or `list`).

    """

    def __init__(self, types: typing.Tuple = (list, tuple)):
        super().__init__()
        self.types = types

    def forward(self, data) -> typing.List[typing.Any]:
        """
        Parameters
        ----------
        data: Iterable[Iterable ... Iterable[Any]]
            Arbitrarily nested data being one of type provided in `types`.
        """
        if not isinstance(data, self.types):
            return data
        return Flatten._flatten(data, self.types)

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
        step ** tt.If(lambda loss: loss ** 10, tt.callbacks.Logger("VERY HIGH LOSS!!!"))

    Parameters
    ----------
    condition: bool | Callable(Any) -> bool
        If boolean value and if `True`, run underlying Operation (or other Callable).
        If Callable, should take data as argument and return decision based on
        that as single `bool`.
    operation: torchtraining.Operation | Callable
        Operation to run if `True`


    Returns
    -------
    Any
        If `true`, returns value from `operation`, otherwise passes original `data`

    """

    def __init__(
        self,
        condition: typing.Union[bool, typing.Callable[[typing.Any], bool,]],
        operation: typing.Callable[[typing.Any,], typing.Any],
    ):
        super().__init__()
        if isinstance(condition, bool):
            self._choice_method = lambda data: condition
        else:
            self._choice_method = lambda data: condition(data)

        self.condition = condition
        self.operation = operation

    def forward(self, data: typing.Any) -> typing.Any:
        """
        Arguments
        ---------
        data: Any
            Anything you want (usually `torch.Tensor` like stuff).
        """
        if self.condition(data):
            return self.operation(data)
        return data

    def __str__(self) -> str:
        if self.condition:
            return str(self.operation)
        return "no-operation"


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

        step ** tt.IfElse(
            lambda loss: loss ** 10,
            tt.callbacks.Logger("VERY HIGH LOSS!!!"),
            tt.callbacks.Logger("LOSS IS NOT THAT HIGH..."),
        )

    Parameters
    ----------
    condition: bool
        Boolean value. If `true`, run underlying Op (or other Callable).
    operation1: torchtraining.Op | Callable
        Operation or callable getting single argument (`data`) and returning anything.
    operation2: torchtraining.Op | Callable
        Operation or callable getting single argument (`data`) and returning anything.


    Returns
    -------
    Any
        If `true`, returns value from `operation1`, otherwise from `operation2`.

    """

    def __init__(
        self,
        condition: bool,
        operation1: typing.Callable[[typing.Any,], typing.Any],
        operation2: typing.Callable[[typing.Any,], typing.Any],
    ):
        super().__init__()
        self.condition = condition
        self.operation1 = operation1
        self.operation2 = operation2

    def forward(self, data: typing.Any) -> typing.Any:
        """
        Arguments
        ---------
        data: Any
            Anything you want (usually `torch.Tensor` like stuff).
        """
        if self.condition:
            return self.operation1(data)
        return self.operation2(data)

    def __str__(self) -> str:
        if self.condition:
            return str(self.operation1)
        return str(self.operation2)


class ToAll(Operation):
    r"""Apply operation to each element of sample.**

    .. note::

        If you want to apply operation to all nested elements (e.g. in nested `tuple`),
        please use `torchtraining.Flatten` object first.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss


        step = TrainStep(criterion, device)

        step ** tt.If(
            lambda loss: loss ** 10,
            tt.callbacks.Logger("VERY HIGH LOSS!!!"),
            tt.callbacks.Logger("LOSS IS NOT THAT HIGH..."),
        )


    Parameters
    ----------
    operation: Callable
        Pipe to apply to each element of sample.

    Returns
    -------
    Tuple[operation(subsample)]
        Tuple consisting of subsamples with operation applied.

    """

    def __init__(self, operation: typing.Callable):
        super().__init__()
        self.operation = operation

    def forward(self, sample):
        """
        Arguments
        ---------
        data: Any
            Anything you want (usually `torch.Tensor` like stuff).
        """
        return tuple(self.operation(subsample) for subsample in sample)


class Lambda(Operation):
    """Run user specified operation on `data`.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return accuracy


        step = TrainStep(criterion, device)

        # If you want to get that SOTA badly, we got ya covered
        step ** tt.Lambda(lambda accuracy: accuracy * 2)

    Parameters
    ----------
    operation: Callable(Any) -> Any
        Single argument callable getting data and returning some value.
    name: str, optional
        `string` representation of this operation (if any).
        Default: `torchtraining.metrics.Lambda`


    Returns
    -------
    Any
        Value returned from `operation`

    """

    def __init__(
        self,
        operation: typing.Callable[[typing.Any,], typing.Any],
        name: str = "torchtraining.Lambda",
    ):
        super().__init__()
        self.operation = operation
        self.name = name

    def __str__(self) -> str:
        return self.name

    def forward(self, data: typing.Any) -> typing.Any:
        """
        Arguments
        ---------
        data: Any
            Anything you want (usually `torch.Tensor` like stuff).
        """
        return self.operation(data)
