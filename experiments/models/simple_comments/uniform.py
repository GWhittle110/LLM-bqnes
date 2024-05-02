"""
Uniform predictions model
"""

import torch
from torch import nn


class Uniform(nn.Module):
    """
    Uniform class predictions
    """

    def __init__(self):
        super(Uniform, self).__init__()

    def forward(self, x):
        return 0.5 * torch.ones_like(x)

