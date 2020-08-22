import importlib

import torch

import horovod.torch as hvd

from .._base import Accelerator
from ..utils import general as utils

if utils.module_exists("horovod.torch"):

    class Horovod(Accelerator):
        """

        Parameters
        ----------
        comm: List, optional
            List specifying ranks for the communicator, relative to the `MPI_COMM_WORLD`
            communicator OR the MPI communicator to use. Given communicator will be duplicated.
            If `None`, Horovod will use MPI_COMM_WORLD Communicator. Default: `None`


        """

        def __init__(
            self,
            model,
            optimizer,
            rank: int = 0,
            per_worker_threads: int = None,
            comm=None,
        ):
            hvd.init(comm)

            if torch.cuda.is_available():
                torch.cuda.set_device(hvd.local_rank())
            if per_worker_threads is not None:
                if per_worker_threads < 1:
                    raise ValueError("Each worker needs at least one thread to run.")
                torch.set_num_threads(per_worker_threads)

            hvd.broadcast_parameters(model.state_dict(), root_rank=rank)
            hvd.broadcast_optimizer_state(optimizer, root_rank=rank)

    from . import horovod
