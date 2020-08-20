"""Cast objects (usually `torch.Tensor`) to specific device.

PyTorch's defaults (as of `1.6.0` CPU and GPU) are forwarded for easier usage.

If you wish to use other (non-standard) devices (like TPUs), please use
`Device` class and explicitly, for example (TPU)::


    import torch_xla.core.xla_model as xm

    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            # Generate loss and other necessary items
            ...
            return loss, predictions, targets


    step = TrainStep(criterion, device)
    # Select `loss` and perform backpropagation
    step > tt.Select(loss=0) > tt.device.Device(xm.xla_device())

Users should use `CPU` class mainly, rest is provided for convenience.

"""

import importlib

import torch

from ._base import Operation


class CPU(Operation):
    """Cast `object` (usually `torch.Tensor`) to `cpu`.

    Example::

        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                return loss, accuracy


        step = TrainStep(criterion, device)
        iteration = tt.iterations.Train(step, module, dataloader)

        # Cast to CPU in order not to inflate GPU memory with `list`
        # You shouldn't use tt.accumulators.List though, just saying
        iteration > tt.Select(
            accuracy=1
        ) > tt.device.CPU() > tt.accumulators.List() > tt.callbacks.Logger("Accuracy")

    Parameters
    ----------
    memory_format: torch.memory_format, optional
        The desired memory format of returned Tensor. Default: torch.preserve_format.
        Default: `torch.preserve_format`

    """

    def __init__(self, memory_format=torch.preserve_format):
        self.memory_format = memory_format

    def forward(self, data):
        return data.cpu(memory_format=self.memory_format)


class CUDA(Operation):
    """Cast `object` (usually `torch.Tensor`) to cuda enabled device.

    Example can be the same as the one presented in `CPU`, though definitely
    this cast shouldn't be used too often.

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
        return data.cuda(
            self.device, self.non_blocking, memory_format=self.memory_format,
        )


class Device(Operation):
    """Cast `object` to any device (for example `TPU` with `torch_xla` package).

    See `example` at the beginning of this section.

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
