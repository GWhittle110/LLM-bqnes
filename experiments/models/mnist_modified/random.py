"""
Uniform predictions model
"""

import torch
from torch import nn


class Random(nn.Module):

    def __init__(self):
        super(Random, self).__init__()

    def forward(self, x):
        preds = torch.rand([len(x), 10], device=x.device)
        preds = (preds.T / preds.sum(dim=1)).T
        return preds

