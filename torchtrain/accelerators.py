import torch

# Setup in __init__
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)


# Context manager
# def cleanup():
#     dist.destroy_process_group()

# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic(rank, world_size):
#     print(f"Running basic DDP example on rank {rank}.")
#     setup(rank, world_size)

#     DDP model has to be passed in ctor
#     # create model and move it to GPU with id rank
#     model = ToyModel().to(rank)
#     ddp_model = DDP(model, device_ids=[rank])

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     optimizer.zero_grad() # ZERO_GRAD handled
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(rank)
#     loss_fn(outputs, labels).backward()
#     optimizer.step()

#     cleanup()


# def run_demo(demo_fn, world_size):
#     mp.spawn(demo_fn,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)


class DistributedDataParallel:
    def __init__(self):
        pass
