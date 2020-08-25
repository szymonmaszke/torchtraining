"""This module provides standard PyTorch operations (like `backward`)
in functional manner.

.. note::

    **IMPORTANT**: This module is used almost all the time
    so be sure to understand how it works.


It allows users to perform training on single `step` for both training and evaluation
using PyTorch's optimizer, backward or zeroing gradient, for example::


    class Step(tt.steps.Step):
        def forward(self, module, sample):
            # Your forward step here
            ...
            return loss, predictions

    training = (
        Step(criterion, gradient=True, device=device)
        ** tt.Select(loss=0)
        ** tt.pytorch.ZeroGrad(network)
        ** tt.pytorch.Backward()
        ** tt.pytorch.Optimize(optimizer)
        ** tt.pytorch.Detach()
    )

    evaluation = (
        Step(criterion, gradient=False, device=device)
        ** tt.Select(predictions=1)
        ** tt.callbacks.Log(writer, "Predicted")
    )

Some other operations are also simplified (e.g. gradient accumulation),
see `torchtraining.callbacks.Optimize`

"""

import torch

from ._base import Operation


class Detach(Operation):
    """Returns a new Tensor, detached from the current graph.

    .. note::

        **IMPORTANT**: This operation should be used before accumulating
        values after `iteration` in order not to grow backpropagation
        graph.

    Returns
    -------
    torch.Tensor
        Detached tensor

    """

    def forward(self, data):
        """
        Arguments
        ---------
        torch.Tensor
            Tensor to be detached (new Tensor is returned).

        """
        return data.detach()


class Schedule(Operation):
    """Run single step of given scheduler.

    Usually placed after each `step` or `iteration` (depending on provided
    scheduler instance).

    Returns
    -------
    torch.Tensor
        Value passed to function initially

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Instance of scheduler-like object with interface aligned with
        `torch.optim.lr_scheduler._LRScheduler` base class
    use_data : bool
        Whether input data should be used when stepping scheduler.
    """

    def __init__(self, scheduler, use_data: bool = False):
        super().__init__()
        self.scheduler = scheduler
        self.use_data = use_data

    def forward(self, data):
        """
        Arguments
        ---------
        torch.Tensor
            Tensor which is optionally used to step scheduler.

        """
        if self.use_data:
            self.scheduler.step(data)
        else:
            self.scheduler.step()

        return data


class Backward(Operation):
    """Run backpropagation on output tensor.

    Parameters
    ----------
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler used for automatic mixed precision mode.
    accumulate: int, optional
        Divide loss by ``accumulate`` if gradient accumulation is used.
        This approach averages gradient from multiple batches.
        Default: `1` (no accumulation)
    gradient : torch.Tensor, optional
        Tensor used as initial value to backpropagation. If unspecified,
        uses `torch.tensor([1.0])` as default value (just like `tensor.backward()` call).

    Returns
    -------
    torch.Tensor
        Tensor after backward (possibly scaled by `accumulate`)

    """

    def __init__(self, scaler=None, accumulate: int = 1, gradient: torch.Tensor = None):
        super().__init__()
        self.scaler = scaler
        self.accumulate = accumulate
        self.gradient = gradient

    def forward(self, data):
        """
        Arguments
        ---------
        data: torch.Tensor
            Tensor on which `backward` will be run (possibly accumulated).
            Usually `loss` value

        """
        output = data / self.accumulate
        if self.scaler is not None:
            output = self.scaler.scale(output)
        if self.gradient is not None:
            output.backward(self.gradient)
        else:
            output.backward()

        return output


class Optimize(Operation):
    """Perform optimization step on `parameters` stored by `optimizer`.

    Currently specifying `closure` and `scaler` is mutually exclusive.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Instance of optimizer-like object with interface aligned with
        `torch.optim.Optimizer`.
    accumulate: int, optional
        Divide loss by ``accumulate`` if gradient accumulation is used.
        This approach averages gradient from multiple batches.
        Default: `1` (no accumulation)
    closure : Callable, optional
        A closure that reevaluates the model and returns the loss.
        Optional for most optimizers. Default: `None`
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler used for automatic mixed precision mode.
        Default: `None`
    *args
        Arguments passed to either `scaler.step` (if specified) or `optimizer.step`
    **kwargs
        Keyword arguments passed to either `scaler.step` (if specified) or `optimizer.step`

    Returns
    -------
    Any
        Anything passed to `forward`.


    """

    def __init__(
        self, optimizer, accumulate: int = 1, closure=None, scaler=None, *args, **kwargs
    ):
        super().__init__()
        self.optimizer = optimizer
        self.accumulate = accumulate

        if scaler is not None and closure is not None:
            raise ValueError("Closure use with scaler is not currently supported.")

        self.scaler = scaler
        self.closure = closure
        self.args = args
        self.kwargs = kwargs

        self._counter = -1

    def forward(self, data):
        """
        Arguments
        ---------
        data: Any
            Anything as it does not influence this operation.

        """
        self._counter += 1
        if self._counter % self.accumulate:
            if self.scaler is not None:
                self.scaler.step(self.optimizer, *self.args, **self.kwargs)
            else:
                self.optimizer.step(self.closure, *self.args, **self.kwargs)

        return data


class ZeroGrad(Operation):
    """Zero model or optimizer gradients.

    Function `zero_grad()` will be run on the provided object.
    Usually called after every `step` (or after multiple steps, see `accumulate`
    argument).

    Parameters
    ----------
    obj : torch.optim.Optimizer | torch.nn.Module
        Instance of object to zero gradient on.
    accumulate : int
        Accumulate gradient for specified number of iterations before zero-ing out
        gradient.

    Returns
    -------
    Any
        Anything passed to `forward`.


    """

    def __init__(self, obj, accumulate: int = 1):
        super().__init__()
        self.obj = obj
        self.accumulate = accumulate
        self._counter = -1

    def forward(self, data):
        """
        Arguments
        ---------
        data: Any
            Anything as it does not influence this operation.

        """
        self._counter += 1
        if self._counter % self.accumulate:
            self.obj.zero_grad()
        return data


class UpdateGradScaler(Operation):
    """Update gradient scaler used with automatic mixed precision.

    Parameters
    ----------
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler used for automatic mixed precision mode.

    Returns
    -------
    Any
        Anything passed to `forward`.


    """

    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, data):
        """
        Arguments
        ---------
        data: Any
            Anything as it does not influence this operation.

        """
        self.scaler.update()
        return data
