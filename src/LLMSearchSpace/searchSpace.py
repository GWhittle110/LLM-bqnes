"""
Search space object definition
"""

import numpy as np
from torch.utils.data import Dataset
from src.utils.logLikelihood import log_likelihood


class SearchSpace:

    def __init__(self, models: list, coordinates: np.ndarray, train_dataset: Dataset = None, log_offset: float = None):
        """
        Keeps track of models, their respective coordinates and likelihoods once calculated. Also manages dataset for
        evaluating likelihoods.
        :param models: List of models
        :param coordinates: Array of coordinates, with order corresponding to that of models
        :param train_dataset: Dataloader for training dataset
        :param log_offset: Offset on log likelihood values. If None, will take the first query result as the offset
        """
        self.coordinates = coordinates
        self.models = models
        self.log_likelihoods = np.nan * np.ones(len(coordinates))
        self.dataset = train_dataset
        self.log_offset = log_offset

    def query_log_likelihood(self, index: int = None, coordinate: np.ndarray = None, batch_size: int = 100) -> float:
        """
        Evaluate model log likelihood on training dataset
        :param index: Index of model to query. Mutually exclusive with coordinate
        :param coordinate: Coordinate of model to query. Mutually exclusive with index
        :param batch_size: batch size for evaluation
        :return: model likelihood
        """

        if not (index is None) ^ (coordinate is None):
            raise ValueError("Must specify exactly one of index and coordinate")

        if coordinate is not None:
            index = np.where((self.coordinates == coordinate).min(axis=1))[0].item()

        model = self.models[index]
        ll = log_likelihood(model, self.dataset, batch_size)

        if self.log_offset is None:
            self.log_offset = ll
        self.log_likelihoods[index] = ll - self.log_offset
        return ll - self.log_offset

            