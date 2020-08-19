"""Special type of callbacks focused on `tensorboard` integration.

Example::

    # Assume iteration was defined and loss is 0th element of step
    iteration = (
        tt.iterations.Iteration(...)
        > tt.Select(loss=0)
        > tt.device.CPU()
        > tt.accumulators.Mean()
        > tt.callbacks.tensorboard.Scalar(writer, "Network/Loss")
    )


"""

import typing

import loguru

from .. import _base


def _docs(data_type, data_description, body=None):
    def wrapper(klass):
        function = klass.__name__.lower()
        docstring = """Log {function} to Tensorboard's summary.

        User should specify single `writer` instance to all `torchtrain.callbacks.tensorboard`
        objects used for training.

        See `torch.utils.tensorboard.writer.add_{function}` for more details.

        Can be used similarly to `torchtrain.callbacks.Logger`

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

        Arguments
        ---------
        data: {data_type}
            {data_description}
        """.format(
            function=function, data_type=data_type, data_description=data_description,
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


@_docs(
    data_type="float | string | torch.Tensor scalar", data_description="Value to save"
)
class Scalar(_Tensorboard):
    pass


@_docs(
    data_type="Dict",
    data_description="Key-value pair storing the tag and corresponding values",
)
class Scalars(_Tensorboard):
    pass


@_docs(
    data_type="torch.Tensor | numpy.array | string/blobname",
    data_description="Values to build histogram",
)
class Histogram(_Tensorboard):
    pass


@_docs(
    data_type="torch.Tensor | numpy.array | string/blobname",
    data_description="Image data",
)
class Image(_Tensorboard):
    pass


@_docs(
    data_type="torch.Tensor | numpy.array | string/blobname",
    data_description=r"""Images data.
    Default shape is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
    accepted. e.g. NCHW or NHWC.
    """,
)
class Images(_Tensorboard):
    pass


@_docs(
    data_type=r"matplotlib.pyplot.figure",
    data_description=r"Figure to render into tensorboard summary",
    body=r"Note that this requires the `matplotlib` package.",
)
class Figure(_Tensorboard):
    pass


@_docs(
    data_type=r"torch.Tensor",
    data_description=r"""Video data of shape :math:`(N, T, C, H, W)`.
    The values should lie in :math:`[0, 255]` for type `uint8` or :math:`[0, 1]` for type float.""",
    body=r"Note that this requires the `moviepy` package.",
)
class Video(_Tensorboard):
    pass


@_docs(
    data_type="torch.Tensor",
    data_description=r"""Sound data.
    Default shape is :math:`(1, L)`. Values should lie between :math:`[-1, 1]`.
    """,
)
class Audio(_Tensorboard):
    pass


@_docs(data_type="string", data_description=r"""String to save.""")
class Text(_Tensorboard):
    pass


@_docs(
    data_type="torch.Tensor",
    data_description=r"""List of the `3D` coordinates of vertices
    of shape  :math:`(B, N, 3)` (`batch`, `number_of_vertices`, `channels`).
    """,
)
class Mesh(_Tensorboard):
    pass


@_docs(
    data_type="Tuple[torch.Tensor | numpy.array | string, torch.Tensor | numpy.array | string]",
    data_description=r"""First element should be neural network predictions (as probability in :math:`[0, 1]` range).
    Second are binary labels :math:`[0, 1]` acting as a ground truth.
    """,
)
class PRCurve(_Tensorboard):
    def forward(self, data):
        predictions, labels = data
        return super().forward(labels, predictions)
