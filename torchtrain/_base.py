import abc
import functools
import inspect
import typing

import yaml


class Base:
    """Common base class for all `torchtrain` objects.

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


class StatefulPipe(Base):
    """Pipe-like component keeping other pipes as internal state.

    Acts as a base for `savers` and `producers` to keep both working
    coherently.

    After using `__or__` same object is returned with right hand side
    added as a sink for values.

    """

    def __init__(self):
        self.pipes = []

    def _apply_pipes(self, data):
        return functools.reduce(
            lambda previous_result, pipe: pipe(previous_result), self.pipes, data,
        )

    def __or__(self, other):
        self.pipes.append(other)
        return self

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def __call__(self, data):
        pass

    @abc.abstractmethod
    def forward(self, data):
        pass

    def __str__(self):
        return yaml.dump({super().__str__(): self.pipes})


class StatelessPipe(Base):
    """Pipe-like component keeping other pipes as internal state.

    Acts as a base for `savers` and `producers` to keep both working
    coherently.

    After using `__or__` same object is returned with right hand side
    added as a sink for values.

    """

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __str__(self) -> str:
        if str(self.second) == "":
            return str(self.first)
        if str(self.first) == "":
            return str(self.second)
        return yaml.dump({str(self.first): str(self.second)})

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        first_result = self.first(data)
        return self.second(first_result)

    def __or__(self, other):
        """Pipe output of this operation into `other`.

        Parameters
        ----------
        other : subclass of torchtrain.Op
            Operation working on output generated by `self`
        """
        if isinstance(other, StatefulPipe):
            return other | self
        return StatelessPipe(self, other)


###############################################################################
#
#                                     OP
#
###############################################################################


class Operation(Base):
    """Base class for operations.

    Usually processes data returned / yielded by
    `torchtrain.steps.Step` / `torchtrain.iterations.Iteration` instances.

    Can also be used with `savers`, in this case those ops are used BEFORE
    data is passed to `saver`.

    Users should implement `forward` method which returns desired value
    after transformation.

    """

    def __call__(self, data) -> typing.Any:
        return self.forward(data)

    @abc.abstractmethod
    def forward(self, data) -> typing.Any:
        pass

    def __or__(self, other):
        """Pipe output of this operation into `other`.

        Parameters
        ----------
        other : subclass of torchtrain.Op
            Operation working on output generated by `self`
        """

        if isinstance(other, StatefulPipe):
            return other | self
        return StatelessPipe(self, other)


###############################################################################
#
#                                    SAVER
#
###############################################################################


class Saver(StatefulPipe):
    """Save values for further reuse.

    Values can be saved after each step or iteration.
    If any pipe is applied with `__or__`, values will be first transformed
    by those pipes and saved as a last step.

    Users should implement `forward` method which returns value(s) which
    are saved after each step. As those values are only references,
    it is a low-cost operation.

    Users should also use `self.data` field in order to accumulate results
    somehow. See `torchtrain.savers` for examples.

    """

    def __init__(self):
        super().__init__()
        self.data = None

    def __call__(self, data):
        self.data = self.forward(self._apply_pipes(data))
        return self.data

    def __str__(self):
        if self.pipes:
            return yaml.dump(list(map(str, self.pipes)) + [super().__str__()])
        return super().__str__()

    @abc.abstractmethod
    def forward(self, data):
        pass


###############################################################################
#
#                                   PRODUCERS
#
###############################################################################


class _ProducerBase(StatefulPipe):
    """Produce values and process/save them.

    This object is unique, as it has `__rshift__` in order to differentiate
    between pipelines leading to saving values instead of merely processing
    them.

    """

    def __init__(self):
        super().__init__()
        self._savers = []
        self._ops = []
        self._save = lambda sample: None

        self.data = None

    # Saving
    def __rshift__(self, saver):
        self._savers.append(saver)
        if len(self._savers) == 1:
            saver = self._savers[0]
            self._save = lambda sample: saver(sample)
        else:
            self._save = lambda sample: tuple([saver(sample) for saver in self._savers])
        return self

    def save(self, *savers):
        for saver in savers:
            self >> saver

        return self

    def __str__(self):
        return {super().__str__(): {"ops": self._ops, "savers": self._savers}}

    # Ops registering
    def __add__(self, other):
        self._ops.append(other)
        return self

    def add(self, *ops: typing.Iterable):
        for op in ops:
            self += op

        return self

    def clear(self):
        for saver in self._savers:
            saver.clear()

        self.data = None

    def feed(self):
        for op in self._ops:
            op(self.data)

    # Convenience context manager
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, *args):
        pass

    # Logic
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Producer(_ProducerBase):
    """Producer returning values via `return` statement.

    Differentiation is needed for smooth user experience
    no matter generators (`iterations` or `epochs`) or simple returns
    (usually `step`).

    """

    def __call__(self, *args, **kwargs):
        with self:
            sample = self._apply_pipes(self.forward(*args, **kwargs))
            self.data = self._save(sample)
            return sample

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class GeneratorProducer(_ProducerBase):
    """Producer yielding values via `yield` statement.

    Differentiation is needed for smooth user experience
    no matter generators (`iterations` or `epochs`) or simple returns
    (usually `step`).

    """

    def __call__(self, *args, **kwargs):
        with self:
            for sample in self.forward(*args, **kwargs):
                sample = self._apply_pipes(sample)
                self.data = self._save(sample)
                yield sample

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
