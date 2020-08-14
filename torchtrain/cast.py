"""Cast tensors in a functional fashion.




"""

import abc

import torch

from ._base import Op


def _docstring(klass) -> str:
    klass.__doc__ = """Cast `torch.Tensor` instance to {cast}.

Other castable types (e.g. `torch.nn.Module`) can be provided as well.

Arguments
---------
data: Any
    Data which will be passed to provided `operations`.

Returns
-------
{cast}
    Casted `data`

""".format(
        cast=klass.__name__
    )
    return klass


class _Cast(Op):
    """Shared base class amongst most casting operations."""

    def __init__(self, memory_format=torch.preserve_format):
        self.memory_format = memory_format

    @abc.abstractmethod
    def forward(self, data):
        pass


@_docstring
class BFloat16(_Cast):
    def forward(self, data):
        return data.bfloat16(self.memory_format)


@_docstring
class Bool(_Cast):
    def forward(self, data):
        return data.bool(self.memory_format)


@_docstring
class Byte(_Cast):
    def forward(self, data):
        return data.byte(self.memory_format)


@_docstring
class Char(_Cast):
    def forward(self, data):
        return data.char(self.memory_format)


@_docstring
class Double(_Cast):
    def forward(self, data):
        return data.double(self.memory_format)


@_docstring
class Float(_Cast):
    def forward(self, data):
        return data.float(self.memory_format)


@_docstring
class Half(_Cast):
    def forward(self, data):
        return data.half(self.memory_format)


@_docstring
class Int(_Cast):
    def forward(self, data):
        return data.int(self.memory_format)


@_docstring
class Long(_Cast):
    def forward(self, data):
        return data.long(self.memory_format)


@_docstring
class Short(_Cast):
    def forward(self, data):
        return data.short(self.memory_format)


@_docstring
class Item(Op):
    def forward(self, data):
        return data.item()


@_docstring
class Numpy(Op):
    def forward(self, data):
        return data.numpy()


@_docstring
class List(Op):
    def forward(self, data):
        return data.to_list()


@_docstring
class MKLDNN(Op):
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
