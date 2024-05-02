"""
Adversarial pytorch linear model
"""

import torch
from torch import nn
import torch.nn.functional as F


class TorchNN2(nn.Module):
    """
    Highly complex neural network implemented in PyTorch
    """
    def __init__(self):
        super(TorchNN2, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.load_state_dict(
            torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\simple\\states\\torchlr.pth'))

    def forward(self, x):
        return F.softmax(-self.fc1(x))