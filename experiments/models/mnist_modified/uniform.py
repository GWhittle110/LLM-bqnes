"""
Uniform predictions model
"""

import torch
from torch import nn


class Uniform(nn.Module):

    def __init__(self):
        super(Uniform, self).__init__()

    def forward(self, x):
        return 0.1 * torch.ones((len(x), 10))

