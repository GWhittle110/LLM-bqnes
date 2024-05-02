"""
Hardcoded boundary linear model
"""

import torch
from torch import nn


class LogisticRegression1(nn.Module):
    """
    Logistic regression
    """

    def __init__(self):
        super(LogisticRegression1, self).__init__()
        self.bias = torch.tensor([0.5, 0.5])
        self.boundary = torch.tensor([1., 1.])

    def forward(self, x):
        boundary = self.boundary.to(x.device)
        bias = self.bias.to(x.device)
        return 0.99 * torch.stack([((x - bias) @ boundary < 0), ((x - bias) @ boundary >= 0)]).T

