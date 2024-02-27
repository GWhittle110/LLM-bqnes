"""
Example model for use in ensemble for MNIST
Shallow mlp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sandbox.mnistEnsembleExample.torchTrain import torchTrain


class MLP1(nn.Module):
    """
    MLP with 4 layers, relu activation functions and slight dropout
    """
    def __init__(self, trained=True):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 10)

        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\mnist_basic\\states\\mlp1.pth'))

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        return F.softmax(x)


if __name__ == "__main__":
    model = MLP1(trained=False)
    torchTrain(model, "mlp1", device=torch.device("cuda:0"), n_epochs=7)

