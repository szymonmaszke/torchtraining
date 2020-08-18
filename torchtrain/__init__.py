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
    """Select output item returned from `step` or `iteration` objects.

    Allows users to focus on specific part of output and pipe this specific
    values to other operations (like metrics, loggers etc.).

    Example::

        import torchtrain as tt


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
        self.output_selection = output_selection.values()

        if len(self.output_selection) == 1:
            self._selection_method = lambda data: data[self.output_selection[0]]
        else:
            self._selection_method = lambda data: [
                data[index] for index in self.output_selection
            ]

    def forward(self, data: typing.Iterable[typing.Any]) -> typing.Any:
        return self._selection_method(data)

    def __str__(self) -> str:
        return yaml.dump({super().__str__(): self.output_selection})


class Split(Operation):
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


class Flatten(Operation):
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


# Make it dynamic and static
class If(Operation):
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


# Make it dynamic and static
class IfElse(Operation):
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


class Drop(_Choice):
    r"""**Return sample without selected elements.**

    Sample has to be indexable object (has `__getitem__` method implemented).

    **Important:**

    - Negative indexing is supported if supported by sample object.
    - This function is **slower** than `Select` and the latter should be preffered.
    - If you want to select sample from nested `tuple`, please use `Flatten` first
    - Returns single element if only one element is left
    - Returns `None` if all elements are dropped

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset | dataset
        # Zeroth and last samples dropped
        selected = new_dataset.map(td.maps.Drop(0, 2))

    Parameters
    ----------
    *indices : int
            Indices of objects to remove from the sample. If left empty, tuple containing
            all elements will be returned.

    Returns
    -------
    Tuple[samples]
            Tuple without selected elements

    """

    def __call__(self, sample):
        return self._magic_unpack(
            tuple(
                sample[index]
                for index, _ in enumerate(sample)
                if index not in self.indices
            )
        )


class ToAll(Base):
    r"""**Apply function to each element of sample.**

    Sample has to be `iterable` object.

    **Important:**

    If you want to apply function to all nested elements (e.g. in nested `tuple`),
    please use `torchdata.maps.Flatten` object first.

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset | dataset
        # Each concatenated sample will be increased by 1
        selected = new_dataset.map(td.maps.ToAll(lambda x: x+1))

    Attributes
    ----------
    function : Callable
            Function to apply to each element of sample.

    Returns
    -------
    Tuple[function(subsample)]
            Tuple consisting of subsamples with function applied.

    """

    def __init__(self, function: typing.Callable):
        self.function = function

    def __call__(self, sample):
        return tuple(self.function(subsample) for subsample in sample)


class To(Base):
    """**Apply function to specified elements of sample.**

    Sample has to be `iterable` object.

    **Important:**

    If you want to apply function to all nested elements (e.g. in nested `tuple`),
    please use `torchdata.maps.Flatten` object first.

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset | dataset
        # Zero and first subsamples will be increased by one, last one left untouched
        selected = new_dataset.map(td.maps.To(lambda x: x+1, 0, 1))

    Attributes
    ----------
    function : Callable
            Function to apply to specified elements of sample.

    *indices : int
            Indices to which function will be applied. If left empty,
            function will not be applied to anything.

    Returns
    -------
    Tuple[function(subsample)]
            Tuple consisting of subsamples with some having the function applied.

    """

    def __init__(self, function: typing.Callable, *indices):
        self.function = function
        self.indices = set(indices)

    def __call__(self, sample):
        return tuple(
            self.function(subsample) if index in self.indices else subsample
            for index, subsample in enumerate(sample)
        )


class Except(Base):
    r"""**Apply function to all elements of sample except the ones specified.**

    Sample has to be `iterable` object.

    **Important:**

    If you want to apply function to all nested elements (e.g. in nested `tuple`),
    please use `torchdata.maps.Flatten` object first.

    Example::

        # Sample-wise concatenate dataset three times
        dataset |= dataset
        # Every element increased by one except the first one
        selected = new_dataset.map(td.maps.Except(lambda x: x+1, 0))

    Attributes
    ----------
    function: Callable
            Function to apply to chosen elements of sample.

    *indices: int
            Indices of objects to which function will not be applied. If left empty,
            function will be applied to every element of sample.

    Returns
    -------
    Tuple[function(subsample)]
            Tuple with subsamples where some have the function applied.

    """

    def __init__(self, function: typing.Callable, *indices):
        self.function = function
        self.indices = set(indices)

    def __call__(self, sample):
        return tuple(
            self.function(subsample) if index not in self.indices else subsample
            for index, subsample in enumerate(sample)
        )


class Lambda(Operation):
    """Run user specified function on `data`.

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
