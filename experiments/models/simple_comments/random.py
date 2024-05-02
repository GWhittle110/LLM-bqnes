"""
Uniform predictions model
"""

import torch
from torch import nn


class Random(nn.Module):
    """
    Random class predictions
    """

    def __init__(self):
        super(Random, self).__init__()

    def forward(self, x):
        preds = torch.rand(len(x), device=x.device)
        return torch.stack((preds, 1-preds)).T

