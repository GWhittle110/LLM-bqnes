"""
Example model for use in ensemble for MNIST
Shallow CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.cifar10 as dataset_module


class ResCNN(nn.Module):
    """
    CNN with residual blocks
    """
    def __init__(self, trained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)

        # Residual block components
        self.rconv1 = nn.Conv2d(12, 16, 3)
        self.rconv2 = nn.Conv2d(16, 20, 3)
        self.rfc = nn.Linear(1280, 1728)

        self.fc1 = nn.Linear(432, 180)
        self.fc2 = nn.Linear(180, 45)
        self.fc3 = nn.Linear(45, 10)

        self.n_blocks = 5


        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\rescnn.pth'))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        for i in range(self.n_blocks):
            dx = F.relu(self.rconv1(x))
            dx = F.relu(self.rconv2(dx))
            dx = torch.flatten(dx, 1)
            dx = F.relu(self.rfc(dx))
            dx = torch.reshape(dx, (-1, 12, 12, 12))
            x = x + dx
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x



if __name__ == "__main__":
    model = ResCNN(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    torch_train(model, train_dataset, test_dataset, "rescnn", device=torch.device("cuda:0"), learning_rate=0.01, momentum=0.9, batch_size_train=64, gamma=0.9, n_epochs=10)

