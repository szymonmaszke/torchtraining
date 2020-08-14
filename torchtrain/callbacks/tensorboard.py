import typing

import loguru

from .. import _base


def _docstring(klass):
    klass.__doc__ = """Log {function} to Tensorboard.

    User should specify single `writer` instance to all `torchtrain.callbacks.tensorboard`
    objects used for training.

    Parameters
    ----------
    writer: torch.utils.tensorboard.SummaryWriter
        Writer responsible for logging values.
    name: str
        Named under which values will be logged into Tensorboard.
    flush: int
        Flushes the event file to disk after `flush` steps.
        Call this method to make sure that all pending events have been written to disk.
    log : str | int, optional
        Severity level for logging object's actions.
        Available levels of logging:
            NONE        0
            TRACE 	5
            DEBUG 	10
            INFO 	20
            SUCCESS 	25
            WARNING 	30
            ERROR 	40
            CRITICAL 	50
        Default: `NONE` (no logging, `0` priority)
    *args
        Variable length arguments passed to `add_{function}` call.
    **kwargs
        Keyword variable length arguments passed to `add_{function}` call.

    """.format(
        function=klass.__name__
    )
    return klass


class _Tensorboard(_base.Op):
    def __init__(
        self,
        writer,
        name: str,
        flush: int = None,
        log: typing.Union[str, int] = "NONE",
        *args,
        **kwargs
    ):
        self.writer = writer
        self.name: str = name
        self.flush = flush
        self.log = log
        self.args = args
        self.kwargs = kwargs
        self._step = -1

    def forward(self, *data):
        self._step += 1
        class_name = type(self).__name__
        getattr(self.writer, "add_{}".format(class_name.lower()))(
            self.name, *data, self._step, *self.args, **self.kwargs
        )
        loguru.log(
            self.log, "{} added to Tensorboard.".format(class_name),
        )
        if (self.flush is not None) and (self.flush % self._step == 0):
            loguru.log(self.log, "Events flushed to disk.")
            self.writer.flush()


@_docstring
class Scalar(_Tensorboard):
    pass


@_docstring
class Scalars(_Tensorboard):
    pass


@_docstring
class Histogram(_Tensorboard):
    pass


@_docstring
class Image(_Tensorboard):
    pass


@_docstring
class Images(_Tensorboard):
    pass


@_docstring
class Figure(_Tensorboard):
    pass


@_docstring
class Video(_Tensorboard):
    pass


@_docstring
class Audio(_Tensorboard):
    pass


@_docstring
class Text(_Tensorboard):
    pass


@_docstring
class Mesh(_Tensorboard):
    pass


@_docstring
class PRCurve(_Tensorboard):
    def forward(self, data):
        labels, predictions = data
        return super().forward(labels, predictions)
