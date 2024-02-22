"""
Calculate log likelihood of model over dataset
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset


def log_likelihood(model, dataset: Dataset, batch_size: int = 100) -> float:
    """
    Calculate log likelihood of model over dataset
    :param model: Predictive model from dataset[inputs] to dataset[targets]
    :param dataset: Dataset to calculate model log likelihood over
    :param batch_size: batch size for evaluation
    :return: Log likelihood
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss = CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        ll = -sum(loss(torch.tensor(model(inputs)), targets).item() for inputs, targets in dataloader)

    return ll


def log_likelihood_from_predictions(predictions, targets) -> float:
    """
    Calculate log likelihood of model predictions
    :param predictions: Predictions of model over dataset
    :param targets: Targets from dataset
    :return: Log likelihood over dataset
    """
    loss = CrossEntropyLoss(reduction='sum')
    ll = -loss(torch.tensor(predictions, requires_grad=False), torch.tensor(targets, requires_grad=False)).item()
    return ll
