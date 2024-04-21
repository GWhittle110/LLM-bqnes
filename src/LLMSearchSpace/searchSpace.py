"""
Search space object definition
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.utils.logLikelihood import log_likelihood, log_likelihood_from_predictions
from src.utils.loadNatsStateDict import load_nats_state_dict


class SearchSpace:

    def __init__(self, models: list, coordinates: np.ndarray, train_dataset: Dataset = None,
                 predictions: pd.DataFrame = None, log_offset: float = None, reduction_factor: float = 1,
                 nats_info: dict = None):
        """
        Keeps track of models, their respective coordinates and likelihoods once calculated. Also manages dataset for
        evaluating likelihoods.
        :param models: List of models
        :param coordinates: Array of coordinates, with order corresponding to that of models
        :param train_dataset: Dataloader for training dataset
        :param predictions: Dataframe containing model predictions keyed on model name and targets
        :param log_offset: Offset on log likelihood values. If None, will take the first query result as the offset
        :param reduction_factor: Factor to divide log likelihood values by (shrinks likelihoods towards 1)
        :param nats_info: Dict containing NATS Bench API, dataset name and architecture indexes
        """
        self.coordinates = coordinates
        self.models = models
        self.log_likelihoods = np.nan * np.ones(len(coordinates))
        self.dataset = train_dataset
        self.predictions = predictions
        self.log_offset = log_offset
        self.reduction_factor = reduction_factor
        self.nats_info = nats_info
        coordinate_tuples = [tuple(coord) for coord in coordinates]
        if nats_info is None:
            self.models_dict = dict(zip(coordinate_tuples, models))
        else:
            self.models_dict = dict(zip(coordinate_tuples, list(zip(models, nats_info["nats_indexes"]))))

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
            try:
                index = np.where((self.coordinates == coordinate).min(axis=1))[0].item()
            except ValueError:
                # Occurs when non unique coordinate indicating the same model, hence take first one
                index = np.where((self.coordinates == coordinate).min(axis=1))[0][0].item()

        if np.isnan(self.log_likelihoods[index]):
            model = self.models[index]
            if self.nats_info is not None:
                api = self.nats_info["api"]
                nats_index = self.nats_info["nats_indexes"][index]
                self.models[index] = load_nats_state_dict(api, model, nats_index, self.nats_info["dataset"])
                ll = -1 * len(self.dataset) * api.get_more_info(nats_index, self.nats_info["dataset"])["train-loss"]
            elif self.predictions is not None:
                ll = log_likelihood_from_predictions(self.predictions[type(model).__name__], self.predictions["Target"])
            else:
                ll = log_likelihood(model, self.dataset, batch_size)

            if self.log_offset is None:
                self.log_offset = ll
            self.log_likelihoods[index] = (ll - self.log_offset) / self.reduction_factor
        return self.log_likelihoods[index]

            