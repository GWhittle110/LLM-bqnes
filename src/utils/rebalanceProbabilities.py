"""
Negative quadrature weights can cause ensemble prediction probabilities outside the interval [0, 1]. In this case we
clamp the negative values to 0 and renormalise
"""

import torch


def rebalance_probabilities(predictions: torch.Tensor) -> torch.Tensor:
    """
    Clamp negative predictions to 0 and renormalise
    :param predictions: Tensor containing prediction probabilities
    :return: Tensor of rebalanced prediction probabilities
    """
    predictions = torch.maximum(predictions, torch.zeros_like(predictions))
    predictions /= predictions.sum(dim=-1).unsqueeze(-1)
    return predictions
