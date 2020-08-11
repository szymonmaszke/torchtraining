"""Cast objects (usually `torch.Tensor`) to specific device.

PyTorch's defaults (as of `1.6.0` CPU and GPU) are forwarded for easier usage.

If you wish to use other (non-standard) devices (like TPUs), please use
`Device` class and explicitly, for example (TPU)::

    impor torchtrain as tt
    import torch_xla.core.xla_model as xm

    device = tt.device.Device(xm.xla_device())


"""

import importlib

import torch

from ._base import Op


class CPU(Op):
    """Cast `object` (usually `torch.Tensor`) to `cpu`.

    Parameters
    ----------
    memory_format: torch.memory_format, optional
        The desired memory format of returned Tensor. Default: torch.preserve_format.
        Default: `torch.preserve_format`

    """

    def __init__(self, memory_format=torch.preserve_format):
        self.memory_format = memory_format

    def forward(self, data):
        return data.cpu(self.memory_format)


class CUDA(Op):
    """Cast `object` (usually `torch.Tensor`) to cuda enabled device.

    Parameters
    ----------
    device: torch.device | int, optional
        Device index to select. It’s a no-op if this argument is a negative integer or None.
        Default: `None`
    non_blocking: bool, optional
        If True and this copy is between CPU and GPU, the copy may occur asynchronously
        with respect to the host. For other cases, this argument has no effect.
        Default: `False`
    memory_format: torch.memory_format, optional
        The desired memory format of returned Tensor. Default: torch.preserve_format.
        Default: `torch.preserve_format`

    """

    def __init__(
        self, device=None, non_blocking=False, memory_format=torch.preserve_format
    ):
        self.device = device
        self.non_blocking = non_blocking
        self.memory_format = memory_format

    def forward(self, data):
        return data.cuda(self.device, self.non_blocking, self.memory_format,)


class Device(Op):
    """Cast `object` to any device (for example `TPU` with `torch_xla` package).

    Parameters
    ----------
    device: torch.device | Any
        Anything which can be used with `torch.Tensor.to` to cast onto
        specified device.

    """

    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data.to(self.device)
