import torchtraining as tt


class AddFakeImages(tt.Operation):
    def __init__(self, dataset):
        self.dataset = dataset

    def forward(self, data):
        self.dataset.add_fake_images(data)
        return data
