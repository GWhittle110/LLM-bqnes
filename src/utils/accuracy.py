"""
Function to compute accuracy over test dataset
"""

import torch
from torch.utils.data import DataLoader, Dataset


def accuracy(model, dataset: Dataset, batch_size: int = 100) -> float:
    """
    Calculate accuracy of model over dataset
    :param model: Predictive model from dataset[inputs] to dataset[targets]
    :param dataset: Dataset to calculate model accuracy over
    :param batch_size: batch size for evaluation
    :return: Log likelihood
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        acc = sum(torch.sum(torch.argmax(model(inputs), dim=1) == targets)
                  for inputs, targets in dataloader) / len(dataset)

    return acc


def accuracy_from_predictions(predictions, targets) -> float:
    """
    Calculate Accuracy of model predictions
    :param predictions: Predictions of model over dataset
    :param targets: Targets from dataset
    :return: Accuracy over dataset
    """
    acc = torch.sum(torch.argmax(torch.tensor(predictions), dim=1) == torch.tensor(targets)) / len(targets)
    return acc.item()

