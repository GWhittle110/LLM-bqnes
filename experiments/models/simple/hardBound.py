"""
Hardcoded boundary linear model
"""

import torch
from torch import nn


class HardBound(nn.Module):

    def __init__(self):
        super(HardBound, self).__init__()
        self.bias = torch.tensor([0.5, 0.5])
        self.boundary = torch.tensor([1., 1.])

    def forward(self, x):
        boundary = self.boundary.to(x.device)
        bias = self.bias.to(x.device)
        return 0.99 * torch.stack([((x - bias) @ boundary < 0), ((x - bias) @ boundary >= 0)]).T

