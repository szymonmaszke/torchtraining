import numbers
import operator
import pathlib
import sys
import typing

import torch

import loguru

from .._base import Op
from . import tensorboard


class Save(Op):
    """
    Save best module according to specified metric.

    Parameters
    ----------
    module : torch.nn.Module
        Module to save.
    path : pathlib.Path
        Path where module will be saved. Usually ends with `.pt` suffix,
        see PyTorch documentation.
    comparator : Callable(Number, Number) -> bool, optional
        Function comparing two values - current metric and best metric.
        If ``true``, save new module and use current value as the best one.
        One can use Python's standard operator library for this argument.
        Default: `operator.gt` (`current` > `best`)
    method : Callable(torch.nn.Module, path) -> None, optional
        Method to save `torch.nn.Module`. Takes module and path
        and returns anything (return value is discarded).
        Might be useful to transform model (located under `module` attribute) into
        `torch.jit.ScriptModule` or do some preprocessing before saving.
        Default: `torch.save` (whole model saving)
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

    def __init__(
        self,
        module: torch.nn.Module,
        path: pathlib.Path,
        comparator: typing.Callable = operator.gt,
        method: typing.Callable = None,
        log: typing.Union[str, int] = "NONE",
    ):
        self.module = module
        self.path = path
        self.comparator = comparator
        self.method = (
            lambda module, path: torch.save(module, path) if method is None else method
        )
        self.log = log

        self.best = None

    def forward(self, data):
        if self.best is None or self.comparator(data, self.best):
            self.best = data
            self.method(self.module, self.path)
            loguru.log(
                self.log, "New best value: {}".format(self.best),
            )

        return data


class EarlyStopping(Op):
    """
    Exit program with status `0` if `patience` was reached without improvement.

    Used to stop training if neural network's desired value didn't improve
    after `patience` steps.

    Parameters
    ----------
    patience : int
        How long not to terminate if metric does not improve
    delta : numbers.Number, optional
        Difference between `best` value and current considered as an improvement.
        Default: `0`.
    comparator : Callable, optional
        Function comparing two values - current metric and best metric.
        If ``true``, reset patience and use current value as the best one.
        One can use Python's standard `operator` library for this argument.
        Default: `operator.gt` (`current` > `best`)
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

    def __init__(
        self,
        patience: int,
        delta: numbers.Number = 0,
        comparator: typing.Callable = operator.gt,
        log="NONE",
    ):
        self.patience = patience
        self.delta = delta
        self.comparator = comparator
        self.log = log
        self.best = None

        self._counter = -1

    def forward(self, data):
        if self.best is None or self.comparator(data, self.best):
            self._counter = -1
        else:
            self._counter += 1
        if self._counter == self.patience:
            loguru.log(self.log, "Stopping early, best found: {}".format(self.best))
            sys.exit(0)


class Unfreeze(Op):
    """
    Unfreeze module's parameters after `n` steps.

    Parameters
    ----------
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler used for automatic mixed precision mode.
    n : int
        Unfreeze module after `n` runs
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

    def __init__(self, module, steps: int = 0, log="NONE"):
        self.module = module
        self.steps = steps
        self.log = log

        self._counter = -1

    def forward(self, data):
        self._counter += 1
        if self._counter == self.steps:
            loguru.log(self.log, "Unfreezing module's parameters")
            for param in self.module.parameters():
                param.requires_grad_(True)
        return data


class Logger(Op):
    """Log data using `loguru`

    Parameters
    ----------
    name : str
        How to name
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
        Default: `INFO`

    """

    def __init__(self, name: str, log="INFO", *args, **kwargs):
        self.name = name
        self.log = log

    def forward(self, data):
        loguru.log(self.log, "{}: {}".format(self.name, data))
        return data
