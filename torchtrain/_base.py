import abc
import functools
import inspect
import typing


class Base:
    def __str__(self) -> str:
        return (
            self.before_repr()
            + f"{type(self).__module__}.{type(self).__name__}"
            + self.after_repr()
        )

    def __repr__(self) -> str:
        parameters = ", ".join(
            "{}={}".format(key, value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )
        return "{}({})".format(self, parameters)

    def after_repr(self):
        return ""

    def before_repr(self):
        return ""

    # Add yaml-like pretty print
    def print(self):
        pass


###############################################################################
#
#                                   PIPES
#
###############################################################################


class StatefulPipe(Base):
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


class StatelessPipe(Base):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __str__(self) -> str:
        if str(self.second) == "":
            return str(self.first)
        if str(self.first) == "":
            return str(self.second)
        return "{} -> {}".format(self.first, self.second)

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


class Op(Base):
    """Base class for operations.

    Processes data returned / yielded by `torchtrain.steps.Step` / `torchtrain.iterations.Iteration`
    instances.

    Users should implement `forward` method.

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
    def __init__(self):
        super().__init__()
        self.data = None

    def __call__(self, data):
        self.data = self.forward(self._apply_pipes(data))
        return self.data

    def before_repr(self):
        pipes = " -> ".join(map(str, self.pipes))
        if pipes != "":
            return "{} -> ".format(pipes)
        return ""

    @abc.abstractmethod
    def forward(self, data):
        pass


###############################################################################
#
#                                   PRODUCERS
#
###############################################################################


class _ProducerBase(StatefulPipe):
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

    def __exit__(self, *args):
        self.feed()
        self.clear()
        return False

    # Textual representation
    def summary(self):
        def format_component(components):
            return "\n".join(
                [
                    "  - {}".format(component)
                    for component in components
                    if str(component) != ""
                ]
            )

        string = "(arguments) -> {} -> (output)".format(self)
        savers = format_component(self._savers)
        if savers:
            starting_point = "(saved)"
            string += " -> {} ->\n{}".format(starting_point, savers)
        else:
            starting_point = "(nothing saved)"
            string += ""
        ops = format_component(self._ops)
        if ops:
            string += "\n{} ->\n{}".format(starting_point, ops)
        return string

    def after_repr(self):
        pipes = " -> ".join(map(str, self.pipes))
        if pipes != "":
            return " -> {}".format(pipes)
        return ""

    # Logic
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Producer(_ProducerBase):
    def __call__(self, *args, **kwargs):
        with self:
            sample = self._apply_pipes(self.forward(*args, **kwargs))
            self.data = self._save(sample)
            return sample

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class GeneratorProducer(_ProducerBase):
    def __call__(self, *args, **kwargs):
        with self:
            for sample in self.forward(*args, **kwargs):
                sample = self._apply_pipes(sample)
                self.data = self._save(sample)
                yield sample

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
