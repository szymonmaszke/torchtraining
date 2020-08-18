import torch

import torchtrain as tt


# Triple update generator, once discriminator
# Lower learning rate (0.0001)
class Generator(tt.steps.Train):
    def forward(self, modules, batch):
        generator, discriminator = modules
        noise, labels = batch
        noise, labels = noise.to(self.device), labels.to(self.device)

        generated_images = generator(noise)
        with torch.no_grad():
            predictions = discriminator(generated_images)
        loss = self.criterion(predictions, labels)

        return loss, predictions, generated_images


class Discriminator(tt.steps.Train):
    def forward(self, module, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        predictions = module(images)
        loss = self.criterion(predictions, labels)

        return loss, predictions, images
