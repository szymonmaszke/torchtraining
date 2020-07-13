import abc

import torch

from ._base import Op


class _Cast(Op):
    def __init__(self, memory_format=torch.preserve_format):
        self.memory_format = memory_format

    @abc.abstractmethod
    def forward(self, data):
        pass


class BFloat16(_Cast):
    def forward(self, data):
        return data.bfloat16(self.memory_format)


class Bool(_Cast):
    def forward(self, data):
        return data.bool(self.memory_format)


class Byte(_Cast):
    def forward(self, data):
        return data.byte(self.memory_format)


class Char(_Cast):
    def forward(self, data):
        return data.char(self.memory_format)


class Double(_Cast):
    def forward(self, data):
        return data.double(self.memory_format)


class Float(_Cast):
    def forward(self, data):
        return data.float(self.memory_format)


class Half(_Cast):
    def forward(self, data):
        return data.half(self.memory_format)


class Int(_Cast):
    def forward(self, data):
        return data.int(self.memory_format)


class Short(_Cast):
    def forward(self, data):
        return data.short(self.memory_format)


class Item(Op):
    def forward(self, data):
        return data.item()


class Numpy(Op):
    def forward(self, data):
        return data.numpy()


class List(Op):
    def forward(self, data):
        return data.to_list()


class MKLDNN(Op):
    def forward(self, data):
        return data.to_mkldnn()


class Sparse(Op):
    def __init__(self, sparse_dims):
        self.sparse_dims = sparse_dims

    def forward(self, data):
        return data.to_sparse(self.sparse_dims)


# As another tensor
class As(Op):
    def __init__(self, tensor):
        self.tensor = tensor

    def forward(self, data):
        return data.type_as(self.tensor)
