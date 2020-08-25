import typing

import torch
from rich import progress

from . import _base, exceptions
from .utils import iterations as iterations_utils


@iterations_utils.docs(
    header="Perform `step` (`train` or `eval`) until `data` is exhausted",
    body="Provided `module` will be passed to every `step`.",
)
class Iteration(_base.Iteration):
    def __init__(
        self,
        step: typing.Any,
        module: torch.nn.Module,
        data: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
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


@iterations_utils.docs(
    header="Perform training step until `data` is exhausted",
    body="Provided `module` will be passed to every `step`.",
)
class Train(Iteration):
    def __init__(
        self,
        step: typing.Any,
        module: torch.nn.Module,
        data: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
        log: typing.Union[int, str] = "NONE",
        *args,
        **kwargs,
    ):
        super().__init__(step, module, data, True, log, *args, **kwargs)


@iterations_utils.docs(
    header="Perform evaluation step until `data` is exhausted",
    body="Provided `module` will be passed to every `step`.",
)
class Eval(Iteration):
    def __init__(
        self,
        step: typing.Any,
        module: torch.nn.Module,
        data: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
        log: typing.Union[int, str] = "NONE",
        *args,
        **kwargs,
    ):
        super().__init__(step, module, data, False, log, *args, **kwargs)


# @utils.iterations.docs(
#     header="Perform evaluation `step`s until `data` is exhausted",
#     body="Provided `module` will be passed to every `step`.",
# )
# class MultiIteration(_base.Iteration):
#     def __init__(
#         self,
#         steps: typing.Iterable[typing.Callable[..., typing.Any]],
#         modules: torch.nn.Module
#         datas: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
#         train: bool,
#         intervals: typing.Optional[typing.Iterable[int]] = None,
#         log: typing.Union[int, str] = "NONE",
#         *args,
#         **kwargs,
#     ):
#         super().__init__()

#         self.steps = steps
#         self.modules = modules
#         self.datas = datas
#         self.train = train

#         if intervals is None:
#             self.intervals = tuple(range(len(steps)))
#         self.intervals = intervals

#         self.log = log
#         self.args = args
#         self.kwargs = kwargs

#     def __call__(self, *args, **kwargs):
#         self.module.train(self.train)
#         with torch.set_grad_enabled(self.train):
#             yield from super().__call__(*args, **kwargs)

#     def forward(self, *args, **kwargs):
#         pass


# @utils.iterations.docs(
#     header="Perform training `step`s until `data` is exhausted",
#     body="Provided `module` will be passed to every `step`.",
# )
# class MultiTrain(MultiIteration):
#     def __init__(
#         self,
#         step: typing.Any,
#         module: torch.nn.Module,
#         data: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
#         log: typing.Union[int, str] = "NONE",
#         *args,
#         **kwargs,
#     ):
#         super().__init__(step, module, data, True, log, *args, **kwargs)


# @utils.iterations.docs(
#     header="Perform evaluation `step`s until `data` is exhausted",
#     body="Provided `module` will be passed to every `step`.",
# )
# class MultiEval(MultiIteration):
#     def __init__(
#         self,
#         step: typing.Any,
#         module: torch.nn.Module,
#         data: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
#         log: typing.Union[int, str] = "NONE",
#         *args,
#         **kwargs,
#     ):
#         super().__init__(step, module, data, False, log, *args, **kwargs)
