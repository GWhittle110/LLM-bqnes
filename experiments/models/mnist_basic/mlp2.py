"""
Example model for use in ensemble for MNIST
Deeper MLP with gelu activation functions instead of relu and slight dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mnistEnsembleExample.torchTrain import torchTrain


class MLP2(nn.Module):
    """
    MLP with 6 layers, gelu activation functions
    """
    def __init__(self, trained=True):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 600)
        self.fc4 = nn.Linear(600, 200)
        self.fc5 = nn.Linear(200, 50)
        self.fc6 = nn.Linear(50, 10)


        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\mnist_basic\\states\\mlp2.pth'))

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = F.gelu(self.fc5(x))
        x = F.gelu(self.fc6(x))
        return F.softmax(x)


if __name__ == "__main__":
    model = MLP2(trained=False)
    torchTrain(model, "mlp2", device=torch.device("cuda:0"), n_epochs=7)

