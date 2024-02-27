"""
Example model for use in ensemble for MNIST
Wider, deeper CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sandbox.mnistEnsembleExample.torchTrain import torchTrain


class CNN2(nn.Module):
    """
    CNN with 2 convolutional layers but more channels, 2 max pools, slight dropout and 3 fully connected layers
    """
    def __init__(self, trained=True):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 30, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(750, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 10)

        if trained:
            self.load_state_dict(torch.load('/sandbox/mnistEnsembleExample/states/cnn2.pth'))
        self.eval()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 750)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.softmax(x)


if __name__ == "__main__":
    model = CNN2(trained=False)
    torchTrain(model, "cnn2", device=torch.device("cuda:0"))

