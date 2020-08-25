"""Exceptions which allow users to finish pipes abruptly.

Usually shouldn't be used except for the above.

"""


class EpochsException(Exception):
    """Special exception caught by `torchtrain.epochs.EpochsBase` objects.

    User should throw this exception if he wants to stop current iteration immediately
    and proceed with the rest of program.

    Should be used with cautious.

    Currently used by `torchtrain.callbacks.EarlyStopping`, `torchtrain.callbacks.TimeStopping`
    and `torchtrain.callbacks.TerminateOnNan`.
    """

    pass


class IterationsException(Exception):
    """Special exception caught by `torchtrain.iterations.IterationsBase` objects.

    User should throw this exception if he wants to stop current iteration immediately
    and proceed with the rest of program.

    Should be used with cautious.

    """

    pass


class EarlyStopping(EpochsException):
    """EarlyStopping special exception"""

    pass


class TimeStopping(EpochsException):
    """TimeStopping special exception"""

    pass


class TerminateOnNan(EpochsException):
    """TerminateOnNaN special exception"""

    pass
