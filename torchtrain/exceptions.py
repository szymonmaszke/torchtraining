class EpochsException(Exception):
    pass


class IterationsException(Exception):
    pass


class EarlyStopping(EpochsException):
    pass


class TimeStopping(EpochsException):
    pass


class TerminateOnNan(EpochsException):
    pass
