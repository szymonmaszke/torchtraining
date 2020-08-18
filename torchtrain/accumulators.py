import abc
import typing

from ._base import Saver


class Last(Saver):
    """Save **only** last state returned passed to it.

    Data from last `step` will be immediately returned.

    `data` **does not** have to implement any concept (has to be simply assignable).

    """

    def __init__(self):
        super().__init__()
        self.data = None

    def clear(self):
        """Assign `None` to `self.data` clearing `saver`"""
        self.data = None

    def forward(self, data):
        self.data = data
        return self.data


class Sum(Saver):
    """Sum data coming into this object.

    Accumulated data will be returned at each step.

    `data` should have `+=` operator implemented between it's instances
    and Python integers.

    """

    def __init__(self):
        super().__init__()
        self.data = 0

    def clear(self):
        """Assign `0` to `self.data` clearing `saver`"""
        self.data = 0

    def forward(self, data):
        self.data += data
        return self.data


class Mean(Saver):
    """Sum data coming into this object.

    Accumulated data will be returned at each step.

    `data` should have `+=` operator implemented between it's instances
    and Python integers.

    """

    def __init__(self):
        super().__init__()
        self.data = 0
        self._counter = 0

    def clear(self):
        """Assign `0` to `self.data` and zero out counter clearing `saver`"""
        self.data = 0
        self._counter = 0

    def forward(self, data: typing.Any):
        self._counter += 1
        self.data += data
        return self.data / self._counter


class List(Saver):
    """Sum data coming into this object.

    List containing data received up to this moment will be returned
    at every `step`.

    `data` **does not** have to implement any concept
    (as it is only appended to `list`).

    """

    def __init__(self):
        super().__init__()
        self.data = []

    def clear(self):
        """Assign empty `list` to `self.data clearing `saver`"""
        self.data = []

    def forward(self, data):
        self.data.append(data)
        return self.data
