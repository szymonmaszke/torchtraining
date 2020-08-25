"""Special type of callbacks focused on `tensorboard` integration.

.. note::
    **IMPORTANT**: Users need `tensorboard` package installed for this
    module to exist.
    You can install it via `pip install torchtraining[tensorboard]`
    (additional libraries for `Image`, `Images`, `Video`, `Figure`)
    will also be installed)
    or install `tensorboard` directly via `pip install -U tensorboard`
    or a-like command (in this case not all functions may be available,
    see PyTorch's `torch.utils.tensorboard.SummaryWriter` docs for
    exact packages needed for each functionality).


Example::

    # Assume iteration was defined and loss is 0th element of step
    iteration = (
        tt.iterations.Iteration(...)
        ** tt.Select(loss=0)
        ** tt.device.CPU()
        ** tt.accumulators.Mean()
        ** tt.callbacks.tensorboard.Scalar(writer, "Network/Loss")
    )


"""

import typing

import loguru

from .. import _base
from ..utils import general as utils


def _docs(body=None):
    def wrapper(klass):
        function = klass.__name__.lower()
        docstring = """Log {function} to Tensorboard's summary.

User should specify single `writer` instance to all `torchtraining.callbacks.tensorboard`
objects used for training.

See `torch.utils.tensorboard.writer.add_{function}` for more details.

Can be used similarly to `torchtraining.callbacks.Logger`

        """.format(
            function=function,
        )
        if body is not None:
            docstring += body

        docstring += """
Parameters
----------
writer: torch.utils.tensorboard.SummaryWriter
    Writer responsible for logging values.
name: str
    Name (tag) under which values will be logged into Tensorboard.
    Can be "/" separated to group values together, e.g. "Classifier/Loss"
    and "Classifier/Accuracy"
flush: int
    Flushes the event file to disk after `flush` steps.
    Call this method to make sure that all pending events have been written to disk.
log : str | int, optional
    Severity level for logging object's actions.
    Available levels of logging:
        * NONE          0
        * TRACE 	5
        * DEBUG 	10
        * INFO 	        20
        * SUCCESS 	25
        * WARNING 	30
        * ERROR 	40
        * CRITICAL 	50
    Default: `NONE` (no logging, `0` priority)
*args
    Variable length arguments passed to `add_{function}` call.
**kwargs
    Keyword variable length arguments passed to `add_{function}` call.

        """.format(
            function=function,
        )
        klass.__doc__ = docstring
        return klass

    return wrapper


class _Tensorboard(_base.Operation):
    def __init__(
        self,
        writer,
        name: str,
        flush: int = None,
        log: typing.Union[str, int] = "NONE",
        *args,
        **kwargs
    ):
        super().__init__()

        self.writer = writer
        self.name: str = name
        self.flush = flush
        self.log = log
        self.args = args
        self.kwargs = kwargs
        self._step = -1

    def forward(self, *data):
        """
        Arguments
        ---------
        data:
            Tensor (or a-like) to be logged into Tensorboard

        """
        self._step += 1
        class_name = type(self).__name__
        getattr(self.writer, "add_{}".format(class_name.lower()))(
            self.name, *data, self._step, *self.args, **self.kwargs
        )
        loguru.logger.log(
            self.log, "{} added to Tensorboard.".format(class_name),
        )
        if (self.flush is not None) and (self.flush % self._step == 0):
            loguru.logger.log(self.log, "Events flushed to disk.")
            self.writer.flush()

        return data


@_docs()
class Scalar(_Tensorboard):
    pass


@_docs()
class Scalars(_Tensorboard):
    pass


@_docs()
class Histogram(_Tensorboard):
    pass


if utils.modules_exist("PIL"):

    @_docs()
    class Image(_Tensorboard):
        pass

    @_docs()
    class Images(_Tensorboard):
        pass


if utils.modules_exist("matplotlib"):

    @_docs(body=r"Note that this requires the `matplotlib` package.",)
    class Figure(_Tensorboard):
        pass


if utils.modules_exist("moviepy"):

    @_docs(body=r"Note that this requires the `moviepy` package.",)
    class Video(_Tensorboard):
        pass


@_docs()
class Audio(_Tensorboard):
    pass


@_docs()
class Text(_Tensorboard):
    pass


@_docs()
class Mesh(_Tensorboard):
    pass


@_docs()
class PRCurve(_Tensorboard):
    def forward(self, data):
        predictions, labels = data
        return super().forward(labels, predictions)
