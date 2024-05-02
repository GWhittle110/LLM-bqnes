"""
Uniform predictions model
"""

import torch
from torch import nn


class LogisticRegression2(nn.Module):
    """
    Logistic regression
    """

    def __init__(self):
        super(LogisticRegression2, self).__init__()

    def forward(self, x):
        preds = torch.rand(len(x), device=x.device)
        return torch.stack((preds, 1-preds)).T

