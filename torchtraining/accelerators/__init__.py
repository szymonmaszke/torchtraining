"""Accelerators enabling distributed (multi-GPU/multi-node) training.

Accelerators should be instantiated only once and used on top-most
module (in the following order):

    * epoch (if exists)
    * iteration (if exists)
    * step

Those are the only objects which can be "piped" into producers, for example::

    tt.accelerators.Horovod(...) ** tt.iterations.Iteration(...)

And should be used in this way (although it's not always necessary).
See `horovod` module for an example.

"""

import importlib

import torch

from .._base import Accelerator
from ..utils import general as utils

if utils.modules_exist("horovod", "horovod.torch"):

    import horovod.torch as hvd

    class Horovod(Accelerator):
        """Accelerate training using Uber's Horovod framework.

        See `torchtraining.accelerators.horovod` package for more information.

        .. note::

            **IMPORTANT**: This object needs `horovod` Python package to be visible.
            You can install it with `pip install -U torchtraining[horovod]`.
            Also you should export `CUDA_HOME` variable like this:
            `CUDA_HOME=/opt/cuda pip install -U torchtraining[horovod]` (your path may vary)


        Parameters
        ----------
        module: torch.nn.Module
            Module to be broadcasted to all processes.
        rank: int, optional
            Root process rank. Default: `0`
        per_worker_threads: int, optional
            Number of threads which can be utilized by each process.
            Default: `pytorch`'s default
        comm: List, optional
            List specifying ranks for the communicator, relative to the `MPI_COMM_WORLD`
            communicator OR the MPI communicator to use. Given communicator will be duplicated.
            If `None`, Horovod will use MPI_COMM_WORLD Communicator.
            Default: `None`

        """

        def __init__(
            self, model, rank: int = 0, per_worker_threads: int = None, comm=None,
        ):
            hvd.init(comm)

            if torch.cuda.is_available():
                torch.cuda.set_device(hvd.local_rank())
            if per_worker_threads is not None:
                if per_worker_threads < 1:
                    raise ValueError("Each worker needs at least one thread to run.")
                torch.set_num_threads(per_worker_threads)

            hvd.broadcast_parameters(model.state_dict(), root_rank=rank)

    from . import horovod
