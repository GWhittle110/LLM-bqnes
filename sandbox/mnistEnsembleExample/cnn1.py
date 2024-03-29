"""
Example model for use in ensemble for MNIST
Shallow CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sandbox.mnistEnsembleExample.torchTrain import torchTrain


class CNN1(nn.Module):
    """
    CNN with 2 convolutional layers, 2 max pools, slight dropout and 2 fully connected layers
    """
    def __init__(self, trained=True):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        if trained:
            self.load_state_dict(torch.load('/sandbox/mnistEnsembleExample/states/cnn1.pth'))
        self.eval()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)


if __name__ == "__main__":
    model = CNN1(trained=False)
    torchTrain(model, "cnn1", device=torch.device("cuda:0"))

