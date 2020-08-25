"""Module containing base classes shared amongst most (all?) classes in `torchtraining`.

Shouldn't be a concern to users, see docs of each if you want to help with
development.

"""

import abc
import functools
import inspect
import typing

import yaml

from . import exceptions


class Base:
    """Common base class for all `torchtraining` objects.

    Defines default `__str__` and `__repr__`.
    Most objects should customize `__str__` according to specific
    needs.

    Custom objects usually use `yaml.dump` to easily see parameters
    and whole pipeline.

    """

    def __str__(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}"

    def __repr__(self) -> str:
        parameters = ", ".join(
            "{}={}".format(key, value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )
        return "{}({})".format(self, parameters)


###############################################################################
#
#                                   PIPES
#
###############################################################################


class Pipe(Base):
    def __init__(self):
        self._pipes = []

    def _pipe(self, data, pipes):
        for pipe in pipes:
            data = pipe(data)
        return data

    def _broadcast(self, data, pipes):
        for pipe in pipes:
            pipe(data)
        return data

    def __pow__(self, other):
        if not isinstance(other, Pipe):
            raise ValueError(
                "Only `torchtraining.Pipe` objects can be used with ** chain operator."
                "Those objects are `torchtraining.Operation`, `torchtraining.iterations.IterationBase`, "
                "`torchtraining.Accumulator`, `torchtraining.steps.StepBase`, `torchtraining.epochs.EpochBase` "
                "or other inheriting from them. Did you mean to inherit from `torchtraining.Operation`?"
            )
        if isinstance(other, Producer):
            raise ValueError("Cannot pipe {} into {} as it's a Producer")
        if isinstance(other, Accumulator):
            if not isinstance(self, Producer):
                return other.add_previous(self)
        self._pipes.append(other)
        return self

    @abc.abstractmethod
    def __call__(self, data) -> typing.Any:
        pass

    @abc.abstractmethod
    def forward(self, data):
        pass


###############################################################################
#
#                                     OP
#
###############################################################################


class Operation(Pipe):
    """Base class for operations.

    Usually processes data returned / yielded by
    `torchtraining.steps.Step` / `torchtraining.iterations.Iteration` instances.

    Can also be used with `accumulators`, in this case those ops are used BEFORE
    data is passed to `accumulator`.

    Users should implement `forward` method which returns desired value
    after transformation.

    `__call__` does nothing, but is kept for API clarity.

    """

    def __call__(self, data) -> typing.Any:
        return self._pipe(self.forward(data), self._pipes)

    @abc.abstractmethod
    def forward(self, data) -> typing.Any:
        pass


###############################################################################
#
#                               ACCUMULATOR
#
###############################################################################


class Accumulator(Pipe):
    """Save values for further reuse.

    Values can be saved after each step or iteration.
    If any pipe is applied with `__gt__`, values will be first transformed
    by those pipes and saved as a last step.

    Users should implement `forward` method which returns value(s) which
    are saved after each step. As those values are only references,
    it is a low-cost operation.

    Users should also use `self.data` field in order to accumulate results
    somehow. See `torchtraining.accumulators` for examples.

    """

    def __init__(self):
        super().__init__()

        self.data = None
        self._previous = []

    def broadcast(self):
        """Run operations added AFTER accumulator.

        Used by `iterations` in order not to run `tensorboard` etc.
        at each value.

        """
        return self._broadcast(self.calculate(), self._pipes)

    def add_previous(self, pipe):
        self._previous.append(pipe)
        return self

    def __call__(self, data):
        return self.forward(self._pipe(data, reversed(self._previous)))

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def calculate(self) -> typing.Any:
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class Accelerator:
    """Base accelerator class.

    Currently IS NOT involved in any pipe'ing or operations
    (though this might be subject to change).

    Supports following as a syntactic sugar::

        accelerator ** step

    Always returns the passed object unchanged

    """

    def __pow__(self, other):
        return other


###############################################################################
#
#                                   PRODUCERS
#
###############################################################################


class Producer(Pipe):
    """Produce values and process/save them."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class ReturnProducer(Producer):
    """Producer returning values via `return` statement.

    Differentiation is needed for smooth user experience
    no matter generators (`iterations` or `epochs`) or simple returns
    (usually `step`).

    Applies pipes added via "**" and return UNMODIFIED data.

    """

    def __pow__(self, other):
        if isinstance(other, Accumulator):
            raise ValueError("You cannot accumulate values from steps.")
        return super().__pow__(other)

    def __call__(self, *args, **kwargs):
        data = self.forward(*args, **kwargs)
        self._broadcast(data, self._pipes)
        return data

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class GeneratorProducer(Producer):
    """Producer yielding values via `yield` statement.

    Differentiation is needed for smooth user experience
    no matter generators (`iterations` or `epochs`) or simple returns
    (usually `step`).

    Applies pipes added via "**" and yields UNMODIFIED data at each step.

    """

    def _reset_accumulators(self):
        for pipe in self._pipes:
            if isinstance(pipe, Accumulator):
                pipe.reset()

    def _broadcast_accumulators(self):
        for pipe in self._pipes:
            if isinstance(pipe, Accumulator):
                pipe.broadcast()

    def __call__(self, *args, **kwargs):
        for data in self.forward(*args, **kwargs):
            self._broadcast(data, self._pipes)
            yield data
        # Broadcast accumulators
        self._broadcast_accumulators()
        # Reset accumulators
        self._reset_accumulators()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Epoch(GeneratorProducer):
    """Base class for epoch-like structures.

    Works like GeneratorProducer, but handles exceptions.EpochsException
    in case earlier exit is required.

    Applies pipes added via "**" and yields UNMODIFIED data at each step.

    """

    def __call__(self, *args, **kwargs):
        with self:
            yield from super().__call__(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, _, exc_val, __):
        if isinstance(exc_val, exceptions.EpochsException):
            return True
        return False

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        for _ in self(*args, **kwargs):
            pass


class Iteration(GeneratorProducer):
    """Base class for iteration-like structures.

    Works like GeneratorProducer, but handles exceptions.IterationsException
    in case earlier exit is required from `iteration`.

    Applies pipes added via "**" and yields UNMODIFIED data at each step.

    """

    def __call__(self, *args, **kwargs):
        with self:
            yield from super().__call__(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, _, exc_val, __):
        if isinstance(exc_val, exceptions.IterationsException):
            return True
        return False

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Step(ReturnProducer):
    """Base class for step-like structures.

    Works like ReturnProducer, is here to provide unified API across
    `steps`, `iterations` and `epochs`.
    """

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
