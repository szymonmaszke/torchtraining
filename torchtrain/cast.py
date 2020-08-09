"""Cast tensors in a functional fashion.




"""

import abc

import torch

from ._base import Op


class _Cast(Op):
    """Shared base class amongst most casting operations."""

    def __init__(self, memory_format=torch.preserve_format):
        self.memory_format = memory_format

    @abc.abstractmethod
    def forward(self, data):
        pass


class BFloat16(_Cast):
    """Cast `torch.Tensor` instance to Google's `bfloat16` format."""

    def forward(self, data):
        return data.bfloat16(self.memory_format)


class Bool(_Cast):
    """Cast `torch.Tensor` instance to `bool`."""

    def forward(self, data):
        return data.bool(self.memory_format)


class Byte(_Cast):
    """Cast `torch.Tensor` instance to `byte` (`uint8`)."""

    def forward(self, data):
        return data.byte(self.memory_format)


class Char(_Cast):
    """Cast `torch.Tensor` instance to `char` (`int8`)."""

    def forward(self, data):
        return data.char(self.memory_format)


class Double(_Cast):
    """Cast `torch.Tensor` instance to `double` (`float64`)."""

    def forward(self, data):
        return data.double(self.memory_format)


class Float(_Cast):
    """Cast `torch.Tensor` instance to `float32`."""

    def forward(self, data):
        return data.float(self.memory_format)


class Half(_Cast):
    """Cast `torch.Tensor` instance to `half` (`float16`)."""

    def forward(self, data):
        return data.half(self.memory_format)


class Int(_Cast):
    """Cast `torch.Tensor` instance to `int32`."""

    def forward(self, data):
        return data.int(self.memory_format)


class Long(_Cast):
    """Cast `torch.Tensor` instance to `long` (`int64`)."""

    def forward(self, data):
        return data.long(self.memory_format)


class Short(_Cast):
    """Cast `torch.Tensor` instance to `short` (`int16`)."""

    def forward(self, data):
        return data.short(self.memory_format)


class Item(Op):
    """Cast `0/1` dimensional single value `torch.Tensor` to it's Python counterpart."""

    def forward(self, data):
        return data.item()


class Numpy(Op):
    """Cast `torch.Tensor` to `numpy.array`."""

    def forward(self, data):
        return data.numpy()


class List(Op):
    """Cast `torch.Tensor` to Python's `list`."""

    def forward(self, data):
        return data.to_list()


class MKLDNN(Op):
    """Cast `torch.Tensor` to MKLDNN format."""

    def forward(self, data):
        return data.to_mkldnn()


class Sparse(Op):
    """Cast `torch.Tensor` to sparse format.

    Parameters
    ----------
    sparse_dims: int, optional
        The number of sparse dimensions to include in the new sparse tensor.
        Default: `None`.

    """

    def __init__(self, sparse_dims=None):
        self.sparse_dims = sparse_dims

    def forward(self, data):
        return data.to_sparse(self.sparse_dims)


# As another tensor
class As(Op):
    """Cast `torch.Tensor` to the same type as `other`.

    Parameters
    ----------
    other: torch.Tensor
        Tensor according to which incoming tensor will be casted.

    """

    def __init__(self, other):
        self.other = other

    def forward(self, data):
        return data.type_as(self.other)


###############################################################################
#
#                               TYPE ALIASES
#
###############################################################################

UInt8 = Byte
Int8 = Char
Int16 = Short
Int32 = Int
Int64 = Long
Float16 = Half
Float32 = Float
Float64 = Double
