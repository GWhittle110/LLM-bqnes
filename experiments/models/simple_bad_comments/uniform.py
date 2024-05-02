"""
Uniform predictions model
"""

import torch
from torch import nn


class LogisticRegression3(nn.Module):
    """
    Logistic regression
    """

    def __init__(self):
        super(LogisticRegression3, self).__init__()

    def forward(self, x):
        return 0.5 * torch.ones_like(x)

