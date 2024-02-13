"""
Function to compute accuracy over test dataset
"""

import torch
from torcheval.metrics import MulticlassAccuracy
from torch.utils.data import DataLoader, Dataset


def accuracy(model, dataset: Dataset, batch_size: int = 100) -> float:
    """
    Calculate log likelihood of model over dataset
    :param model: Predictive model from dataset[inputs] to dataset[targets]
    :param dataset: Dataset to calculate model log likelihood over
    :param batch_size: batch size for evaluation
    :return: Log likelihood
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        acc = sum(torch.count_nonzero(torch.argmax(model(inputs), dim=1) == targets)
                  for inputs, targets in dataloader) / len(dataset)

    return acc
