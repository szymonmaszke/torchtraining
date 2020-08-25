import torch
import torchtraining as tt
import torchvision

from . import data, iterations, modules, steps


def test_training():
    device = torch.device("cpu")

    dataloaders = data.get(batch_size=8)

    criterion = torch.nn.CrossEntropyLoss()
    network = modules.Classifier(in_channels=3, labels=100)
    optimizer = torch.optim.Adam(network.parameters())

    train_step, eval_step = (
        steps.train(optimizer, criterion, device),
        steps.eval(criterion, device),
    )

    writer = torch.utils.tensorboard.SummaryWriter("log")
    train_iteration, validation_iteration, test_iteration = (
        iterations.train(
            writer, train_step, network, dataloaders["train"], name="Train"
        ),
        iterations.eval(
            writer, eval_step, network, dataloaders["validation"], name="Validation"
        ),
        iterations.eval(writer, eval_step, network, dataloaders["test"], name="Test"),
    )

    epochs = tt.epochs.Epoch(
        train_iteration, validation_iteration, test_iteration, epochs=2
    )
    epochs.run()
