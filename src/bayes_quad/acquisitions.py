"""
Acquisition functions for use with Bayesian quadrature
"""

import numpy as np
from src.bayes_quad.gp import GP, SqWarpedGP
from abc import ABC
from src.LLMSearchSpace.searchSpace import SearchSpace
import logging
from pandas import DataFrame


class Acquisition(ABC):
    """
    Abstract Base class of acquisition functions
    """
    def __init__(self, gp: GP, search_space: SearchSpace):
        """
        :param gp: Underlying GP model
        :param search_space: Search space from which to acquire
        """
        self.gp = gp
        self.search_space = search_space
        self.logger = logging.getLogger("Acquisition")

    def eval(self, x) -> float:
        """
        Evaluate acquisition metric at this coordinate. Must be implemented
        :param x: Query point
        :return: Acquisition metric value at query point
        """
        raise NotImplementedError

    def next(self) -> np.ndarray:
        """
        Calculate the next point to be acquired
        :return: Coordinates of next point to be acquired
        """
        raise NotImplementedError

    def acquire(self, n, batch_size: int = 100) -> None:
        """
        Sequentially acquire and condition gp on n points
        :param n: Number of points to acquire
        :param predictions: Dataframe containing model predictions keyed on model name and targets
        :param batch_size: Batch size for evaluating likelihoods
        :return: None
        """
        for i in range(n):
            next_x = self.next()
            next_y = np.exp(self.search_space.query_log_likelihood(coordinate=next_x,
                                                                   batch_size=batch_size))
            self.gp.add_data(next_x, next_y)


class DiscreteUncertaintySampling(Acquisition):

    def eval(self, x):
        """
        Evaluate acquisition metric (GP uncertainty) at given query point
        :param x: Query point
        :return: GP uncertainty at query point
        """
        _, variance = self.gp.predict(x)
        return variance

    def next(self) -> np.ndarray:
        """
        Calculate candidate point giving rise to maximum GP uncertainty
        :return: Candidate point corresponding to maximum GP uncertainty
        Example, standard GP:
        >>> logging.basicConfig(level=20)
        >>> x = np.random.rand(2).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> surrogate = GP(x, y)
        >>> surrogate.plot()
        >>> candidates = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> search_space = SearchSpace([None], candidates)
        >>> acquisition = DiscreteUncertaintySampling(surrogate, search_space)
        >>> for i in range(5):
                next_x = acquisition.next()
                next_y = np.sin(np.pi*next_x).reshape(-1)
                surrogate.add_data(next_x, next_y)
                surrogate.plot()
        Example, square root warped GP:
        >>> x = np.random.rand(2).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> surrogate = SqWarpedGP(x, y)
        >>> surrogate.plot()
        >>> candidates = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> search_space = SearchSpace([None], candidates)
        >>> acquisition = DiscreteUncertaintySampling(surrogate, search_space)
        >>> for i in range(10):
                next_x = acquisition.next()
                next_y = np.sin(np.pi*next_x).reshape(-1)
                print(f"Acquired point {next_x}. Corresponding function evalution: {next_y}")
                surrogate.add_data(next_x, next_y)
                surrogate.plot()
        """
        candidates = np.array(list(set(map(tuple, self.search_space.coordinates)) - set(map(tuple, self.gp.x))))
        evaluations = self.eval(candidates)
        self.logger.info(f"Acquired coordinate: {candidates[np.argmax(evaluations)]}")
        return candidates[np.argmax(evaluations)]

