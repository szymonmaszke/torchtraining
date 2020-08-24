"""This module allows user to run multiple `iterations` in consecutive order,
which constitutes a single epoch.



"""

import abc
import contextlib
import typing

import loguru

from . import _base, exceptions


class Epoch(_base.Epoch):
    """
    Loop over specified `iterations` until `epochs` number is reached.

    Parameters
    ----------
    epochs : int
        How many epochs should be run
    iterations : Iterable[torchtrain.iterations.Iteration]
        Iterations to be run one after another during single loop pass.
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

    def __init__(self, *iterations, epochs: int, log: typing.Union[str, int] = "NONE"):
        super().__init__()
        self.iterations = iterations
        self.epochs = epochs
        self.log = log

    def __len__(self):
        return self.epochs

    def forward(self, *args, **kwargs):
        for index in range(self.epochs):
            loguru.logger.log(self.log, "Starting epoch {}.".format(index))
            for iteration in self.iterations:
                yield from iteration(*args, **kwargs)

        loguru.logger.log(self.log, "Finished epochs.")
