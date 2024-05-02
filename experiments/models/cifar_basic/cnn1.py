"""
Example model for use in ensemble for MNIST
Shallow CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.cifar10 as dataset_module


class CNN1(nn.Module):
    """
    CNN with 2 convolutional layers, no dropout and 3 fully connected layers
    """
    def __init__(self, trained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\cnn1.pth'))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x



if __name__ == "__main__":
    model = CNN1(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    torch_train(model, train_dataset, test_dataset, "cnn1", device=torch.device("cuda:0"), learning_rate=0.01, momentum=0.9, batch_size_train=64, gamma=0.9, n_epochs=20)

