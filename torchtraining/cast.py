"""Cast tensors in a functional fashion.

Users can use this module to cast `step` outputs to desired type or
to lower precision in order to save memory (though it shouldn't be needed.)

"""

import abc

import torch

from ._base import Operation


def _docstring(klass) -> str:
    klass.__doc__ = """Cast `torch.Tensor` instance to {cast}.

.. note::

    **IMPORTANT**: Only `torch.Tensor` can be passed as `memory_format`
    is specified during casting.


Returns
-------
{cast}
    Casted `data`

""".format(
        cast=klass.__name__
    )
    return klass


def _forward_docstring(function):
    function.__doc__ = """
    Arguments
    ---------
    data: torch.Tensor
        Tensor to be casted
    """
    return function


class _Cast(Operation):
    """Shared base class amongst most casting operations."""

    def __init__(self, memory_format=torch.preserve_format):
        super().__init__()
        self.memory_format = memory_format

    @abc.abstractmethod
    def forward(self, data):
        pass


@_docstring
class BFloat16(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.bfloat16(memory_format=self.memory_format)


@_docstring
class Bool(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.bool(memory_format=self.memory_format)


@_docstring
class Byte(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.byte(memory_format=self.memory_format)


@_docstring
class Char(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.char(memory_format=self.memory_format)


@_docstring
class Double(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.double(memory_format=self.memory_format)


@_docstring
class Float(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.float(memory_format=self.memory_format)


@_docstring
class Half(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.half(memory_format=self.memory_format)


@_docstring
class Int(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.int(memory_format=self.memory_format)


@_docstring
class Long(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.long(memory_format=self.memory_format)


@_docstring
class Short(_Cast):
    @_forward_docstring
    def forward(self, data):
        return data.short(memory_format=self.memory_format)


@_docstring
class Item(Operation):
    @_forward_docstring
    def forward(self, data):
        return data.item()


@_docstring
class Numpy(Operation):
    @_forward_docstring
    def forward(self, data):
        return data.numpy()


@_docstring
class List(Operation):
    @_forward_docstring
    def forward(self, data):
        return data.to_list()


@_docstring
class MKLDNN(Operation):
    @_forward_docstring
    def forward(self, data):
        return data.to_mkldnn()


class Sparse(Operation):
    """Cast `torch.Tensor` to sparse format.

    Parameters
    ----------
    sparse_dims: int, optional
        The number of sparse dimensions to include in the new sparse tensor.
        Default: `None`.

    """

    def __init__(self, sparse_dims=None):
        super().__init__()
        self.sparse_dims = sparse_dims

    @_forward_docstring
    def forward(self, data):
        return data.to_sparse(sparse_dims=self.sparse_dims)


# As another tensor
class As(Operation):
    """Cast `torch.Tensor` to the same type as `other`.

    Parameters
    ----------
    other: torch.Tensor
        Tensor according to which incoming tensor will be casted.

    """

    def __init__(self, other):
        super().__init__()
        self.other = other

    @_forward_docstring
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
