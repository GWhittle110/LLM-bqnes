"""
Calculate log likelihood of model over dataset
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset


def log_likelihood(model, dataset: Dataset, batch_size: int = 25, batch_likelihood: bool = True) -> float:
    """
    Calculate log likelihood of model over dataset
    :param model: Predictive model from dataset[inputs] to dataset[targets]
    :param dataset: Dataset to calculate model log likelihood over
    :param batch_size: batch size for evaluation
    :param batch_likelihood: whether to average likelihoods over batches for final calculation
    :return: Log likelihood
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss = CrossEntropyLoss(reduction='mean' if batch_likelihood else 'sum')

    with torch.no_grad():
        ll = -sum(loss(model(inputs), targets).item() for inputs, targets in dataloader)

    return ll
