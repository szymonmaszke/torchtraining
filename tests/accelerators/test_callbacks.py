import importlib

import torch
import torchtraining as tt

if importlib.util.find_spec("horovod") is not None:

    def test_smoke():
        model = torch.nn.Linear(20, 10)
        accelerator = tt.accelerators.Horovod(model)
        optimizer = tt.accelerators.horovod.optimizer(
            torch.optim.Adam(model.parameters()),
            named_parameters=model.named_parameters(),
        )
