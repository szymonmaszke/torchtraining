import typing

import loguru

from .. import _base


class _Tensorboard(_base.Op):
    """Log {function}.

    User should specify single `writer` instance to all `torchtrain.callbacks.tensorboard`
    objects.

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
            Writer responsible for logging values.
    name : str
            Named under which values will be logged into Tensorboard.
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

    """

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


class Scalar(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("scalar")


class Scalars(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("scalars")


class Histogram(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("histogram")


class Image(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("image")


class Images(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("image")


class Figure(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("figure")


class Video(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("video")


class Audio(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("audio")


class Text(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("text")


class Mesh(_Tensorboard):
    __doc__ = _Tensorboard.__doc__.format("mesh")

class PRCurve(_Tensorboard)
    __doc__ = _Tensorboard.__doc__.format("pr_curve")

    def forward(self, data):
        labels, predictions = data
        return super().forward(labels, predictions)

# To add - graph, embedding, hparams
