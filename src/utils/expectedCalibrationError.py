"""
Calculate expected calibration error of model over dataset
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def expected_calibration_error(model, dataset: Dataset, nbins: int = 10, batch_size: int = 100) -> float:
    """
    Calculate log likelihood of model over dataset
    :param model: Predictive model from dataset[inputs] to dataset[targets]
    :param dataset: Dataset to calculate model log likelihood over
    :param nbins: Number of bins to calculate over
    :param batch_size: batch size for evaluation
    :return: Log likelihood
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        predictions = torch.cat([model(inputs) for inputs, targets in dataloader]).cpu().numpy()

    return expected_calibration_error_from_predictions(predictions, dataset.targets.numpy(), nbins)


def expected_calibration_error_from_predictions(predictions, targets, nbins: int = 10) -> float:
    """
    Calculate log likelihood of model predictions
    :param predictions: Predictions of model over dataset
    :param targets: Targets from dataset
    :param nbins: Number of bins to calculate over
    :return: Log likelihood over dataset
    """
    bin_boundaries = np.linspace(0, 1, nbins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(predictions, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(predictions, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label == targets

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece
