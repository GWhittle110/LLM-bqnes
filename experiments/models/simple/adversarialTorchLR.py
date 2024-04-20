"""
Adversarial pytorch linear model
"""

import torch
from torch import nn
import torch.nn.functional as F


class AntiTorchLR(nn.Module):
    def __init__(self):
        super(AntiTorchLR, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.load_state_dict(
            torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\simple\\states\\torchlr.pth'))

    def forward(self, x):
        return F.softmax(-self.fc1(x))