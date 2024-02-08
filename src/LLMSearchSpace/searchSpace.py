"""
Search space object definition
"""

import numpy as np
import torch


class SearchSpace:

    def __init__(self, models: list, coordinates: np.ndarray, train_dataset: torch.utils.data.DataLoader = None):
        """
        Keeps track of models, their respective coordinates and likelihoods once calculated. Also manages dataset for
        evaluating likelihoods.
        :param models: List of models
        :param coordinates: Array of coordinates, with order corresponding to that of models
        :param train_dataset: Dataloader for training dataset
        """
        self.coordinates = coordinates
        self.models = models
        self.likelihoods = np.array([None] * len(coordinates))
        self.dataset = train_dataset

    def query_likelihood(self, index: int = None, coordinate: np.ndarray = None, batch_size: int = 25,
                         batch_likelihood: bool = True):
        """
        Evaluate model likelihood
        :param index: Index of model to query. Mutually exclusive with coordinate
        :param coordinate: Coordinate of model to query. Mutually exclusive with index
        :param batch_size: batch size for evaluating
        :param batch_likelihood: whether to average likelihoods over batches for final calculation
        :return: model likelihood
        """

        if not (index is None) ^ (coordinate is None):
            raise ValueError("Must specify exactly one of index and coordinate")

        if coordinate is not None:
            index = np.where(self.coordinates == coordinate)
            