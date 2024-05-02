"""
Pytorch linear model
"""

import torch
from torch import nn
import torch.nn.functional as F
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.logistic as dataset_module


train_dataset = getattr(dataset_module, "train_dataset")
test_dataset = getattr(dataset_module, "test_dataset")


class TorchLR(nn.Module):

    def __init__(self, trained=True):
        super(TorchLR, self).__init__()
        self.fc1 = nn.Linear(2, 2)

        if trained:
            self.load_state_dict(
                torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\simple\\states\\torchlr.pth'))

    def forward(self, x):
        return F.softmax(self.fc1(x))


if __name__ == "__main__":
    model = TorchLR(trained=False)
    torch_train(model, train_dataset, test_dataset, "torchlr", device=torch.device("cuda:0"), n_epochs=3)