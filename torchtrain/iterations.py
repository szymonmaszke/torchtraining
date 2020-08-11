import typing

import torch

from rich import progress

from . import _base


def _docstring(header, body, add_train: bool = False):
    body = r"""{}.

    {}

    Parameters
    ----------
    step: torchtrain.steps.Step
        Single step to run. Usually subclass of `torchtrain.steps.Step`, but could be
        any `Callable` taking `module` and `data` arguments and returning anything.
    module : torch.nn.Module
        Torch module (or modules) passed to `step` during each call.
    data : [torch.utils.data.Dataset | torch.utils.data.DataLoader]
        Iterable object (usually data or dataloader) yielding data passed
        to `step`.
    """.format(
        header, body
    )

    if add_train:
        body += r"""
        train: bool
            Whether `module` should be in training state (`module.train()`)
            with enabled gradient or in evaluation mode (`module.eval()`) with
            disabled gradient
        """

    return (
        body
        + r"""
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

    """
    )


class Iteration(_base.GeneratorProducer):
    __doc__ = _docstring(
        header="Perform training `step`s until `data` is exhausted",
        body="Provided `module` will be passed to every `step`.",
        add_train=True,
    )

    def __init__(
        self,
        step: typing.Any,
        module: torch.nn.Module,
        data: torch.utils.data.Dataset,
        train: bool,
        log: typing.Union[int, str] = "NONE",
        *args,
        **kwargs,
    ):
        super().__init__()

        self.step = step
        self.module = module
        self.data = data
        self.train = train

        self.log = log
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self.module.train(self.train)
        with torch.set_grad_enabled(self.train):
            yield from super().__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        for sample in progress.track(
            self.data, description=" ", *self.args, **self.kwargs
        ):
            yield self.step(self.module, sample, *args, **kwargs)


class Train(Iteration):
    __doc__ = _docstring(
        header="Perform training `step`s until `data` is exhausted",
        body="Provided `module` will be passed to every `step`.",
    )

    def __init__(
        self,
        step: typing.Any,
        module: torch.nn.Module,
        data: torch.utils.data.Dataset,
        log: typing.Union[int, str] = "NONE",
        *args,
        **kwargs,
    ):
        super().__init__(step, module, data, True, log, *args, **kwargs)


class Eval(Iteration):
    __doc__ = _docstring(
        header="Perform evaluation `step`s until `data` is exhausted",
        body="Provided `module` will be passed to every `step`.",
    )

    def __init__(
        self,
        step: typing.Any,
        module: torch.nn.Module,
        data: torch.utils.data.Dataset,
        log: typing.Union[int, str] = "NONE",
        *args,
        **kwargs,
    ):
        super().__init__(step, module, data, False, log, *args, **kwargs)
