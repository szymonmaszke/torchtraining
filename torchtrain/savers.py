import abc

from ._base import Saver


# State never changes
class Last(Saver):
    def __init__(self):
        super().__init__()
        self.data = None

    def clear(self):
        self.data = None

    def forward(self, data):
        self.data = data
        return self.data


class Sum(Saver):
    def __init__(self):
        super().__init__()
        self.data = 0

    def clear(self):
        self.data = 0

    def forward(self, data):
        self.data += data
        return self.data


class Mean(Saver):
    def __init__(self):
        super().__init__()
        self.data = 0
        self._counter = 0

    def clear(self):
        self.data = 0
        self._counter = 0

    def forward(self, data):
        self._counter += 1
        self.data += data
        return self.data / self._counter


class List(Saver):
    def __init__(self):
        super().__init__()
        self.data = []

    def clear(self):
        self.data = []

    def forward(self, data):
        self.data.append(data)
        return self.data
