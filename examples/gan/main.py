import operator

import torch
import torchvision

import datasets
import modules
import operations
import steps
import torchtraining as tt


def prepare_generator(writer, dataset, device):
    generator = modules.Generator(in_channels=3)
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.0001)
    step = (
        steps.Generator(tt.loss.SmoothBinaryCrossEntropy(alpha=0.1), device)
        > tt.Select(loss=0)
        > tt.pytorch.ZeroGrad(optimizer)
        > tt.pytorch.Backward()
        > tt.pytorch.Optimize(optimizer)
        > tt.pytorch.Detach()
    )

    step > tt.Select(generated_images=2) > tt.callbacks.tensorboard.Images(
        writer, "Generator/Images"
    ) > operations.AddFakeImages(dataset)

    return generator, step


def prepare_discriminator(device):
    discriminator = modules.Discriminator(in_channels=3)
    optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.0004)
    step = (
        steps.Discriminator(tt.loss.SmoothBinaryCrossEntropy(alpha=0.1), device)
        > tt.Select(loss=0)
        > tt.pytorch.ZeroGrad(optimizer)
        > tt.pytorch.Backward()
        > tt.pytorch.Optimize(optimizer)
        > tt.pytorch.Detach()
    )

    return discriminator, step


def prepare_iteration(
    writer,
    generator,
    generator_step,
    discriminator,
    discriminator_step,
    noise_dataset,
    cifar10_with_fake,
):
    iteration = tt.iterations.MultiIteration(
        steps=(generator_step, discriminator_step),
        modules=((generator, discriminator), discriminator),
        datas=(noise_dataset, cifar10_with_fake),
        intervals=(4, 1),
        log="INFO",
    )

    iteration > tt.Select(loss=0) > tt.device.CPU() > tt.Except(
        tt.accumulators.Mean(), 4
    ) > tt.Split(
        tt.callbacks.tensorboard.Scalar(writer, "Generator/Loss"),
        tt.callbacks.Logger(name="Generator Mean"),
        tt.callbacks.Save(generator, "generator.pt", comparator=operator.lt),
    )
    iteration > tt.Select(loss=0) > tt.device.CPU() > tt.Except(
        tt.accumulators.Mean(), begin=0, end=4
    ) > tt.Split(
        tt.callbacks.tensorboard.Scalar(writer, "Discriminator/Loss"),
        tt.callbacks.Logger(name="Generator Mean"),
        tt.callbacks.Save(discriminator, "generator.pt", comparator=operator.lt),
    )

    return iteration


def main():
    device = torch.device("cuda")
    writer = torch.utils.tensorboard.SummaryWriter("log")

    noise_dataset = datasets.Noise(batch_size=32, features=512)
    cifar10_with_fake = datasets.CIFAR10AndFakeImages(
        torchvision.datasets.CIFAR10(
            "", train=True, transform=torchvision.transforms.ToTensor()
        )
    )
    generator, generator_step = prepare_generator(writer, cifar10_with_fake, device)
    discriminator, discriminator_step = prepare_discriminator(device)
    iteration = prepare_iteration(
        writer,
        generator,
        generator_step,
        discriminator,
        discriminator_step,
        noise_dataset,
        cifar10_with_fake,
    )
    epochs = tt.epochs.Epoch(iteration, 20)
    epochs.run()


if __name__ == "__main__":
    main()
