"""
Acquisition functions for use with Bayesian quadrature
"""

import jax.numpy as jnp
import numpy as np
from src.bayes_quad.gp import GP, SqWarpedGP


class DiscreteUncertaintySampling:

    def __init__(self, gp: GP):
        """
        Propose next point based on maximum uncertainty in integrand. Discrete version.
        :param gp: Underlying GP model
        """
        self.gp = gp

    def eval(self, x):
        """
        Evaluate acquisition metric (GP uncertainty) at given query point
        :param x: Query point
        :return: GP uncertainty at query point
        """
        _, variance = self.gp.predict(x)
        return variance

    def acquire(self, candidates: np.array):
        """
        Calculate candidate point giving rise to maximum GP uncertainty
        :param candidates: candidate points
        :return: Candidate point corresponding to maximum GP uncertainty
        Example, standard GP:
        >>> x = np.random.rand(2).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> surrogate = GP(x, y)
        >>> surrogate.plot()
        >>> acquisition = DiscreteUncertaintySampling(surrogate)
        >>> candidates = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> for i in range(5):
                next_x = acquisition.acquire(candidates)
                next_y = np.sin(np.pi*next_x).reshape(-1)
                print(f"Acquired point {next_x}. Corresponding function evalution: {next_y}")
                surrogate.add_data(next_x, next_y)
                surrogate.plot()
        Example, square root warped GP:
        >>> x = np.random.rand(2).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> surrogate = SqWarpedGP(x, y)
        >>> surrogate.plot()
        >>> acquisition = DiscreteUncertaintySampling(surrogate)
        >>> candidates = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> for i in range(10):
                next_x = acquisition.acquire(candidates)
                next_y = np.sin(np.pi*next_x).reshape(-1)
                print(f"Acquired point {next_x}. Corresponding function evalution: {next_y}")
                surrogate.add_data(next_x, next_y)
                surrogate.plot()
        """
        evaluations = self.eval(candidates)
        return candidates[jnp.argmax(evaluations)]

