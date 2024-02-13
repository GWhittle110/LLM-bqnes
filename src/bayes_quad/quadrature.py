"""
Integrand models equipped with Bayesian quadrature routines
"""

import numpy as np
import tinygp
from src.bayes_quad.gp import GP, SqWarpedGP
from scipy.stats import qmc


class IntegrandModel:

    def __init__(self, x_init: np.ndarray,
                 y_init: np.ndarray,
                 kernel: tinygp.kernels.Kernel = tinygp.kernels.stationary.ExpSquared,
                 theta_init: dict = None,
                 optimize_init: bool = True,
                 surrogate: type = GP):
        """
        Standard Bayesian quadrature
        :param x_init: Input points used to initialise the surrogate GP, n_init X ndim
        :param y_init: Output point used to initialise the surrogate GP, n_init X 1
        :param kernel: Surrogate GP kernel, defaults to RBF Kernel
        :param theta_init: Initial values for kernel hyperparameters. Dict with fields "log_scale", "log_amp" and
        "log_jitter"
        :param optimize_init: If true, optimize hyperparameters on initial data. Forced to False if no initial data
        :param surrogate: Class of surrogate model
        """

        self.surrogate = surrogate(x_init, y_init, kernel, theta_init, optimize_init)
        self.quad_weights = None
        self.quad_points = None
        self.evidence = None
        self.variance = None

    def add_data(self, x, y, *args, **kwargs):
        """
        Add data to surrogate gp
        :param x: np.ndarray, evaluation point
        :param y: np.ndarray, evaluation value
        """
        self.surrogate.add_data(x, y, *args, **kwargs)

    def discrete_quad(self, coords: np.ndarray, prior: np.ndarray = None, min_det: float = 0.001):
        """
        Calculate summation (as an expectation) and quadrature weights for used quadrature points for given coords. Uses
        a constant mean function of the mean of the conditioning points. Also returns uncertainty in summation. To
        compute explicit sum not an expectation use a prior of ones.
        :param min_det: minimum determinant of quadrature point correlation matrix, to stabilise inversion
        :param coords: points to sum over
        :param prior: prior probabilities for each coord
        :return: (summation, variance)
        Example, 1d:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> integrand = IntegrandModel(x, y)
        >>> integrand.surrogate.plot()
        >>> coords = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> val, var = integrand.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> x = np.vstack((x, x))
        >>> y = np.sin(np.pi*x).reshape(-1) + 0.05 * np.random.randn(len(x))
        >>> integrand = IntegrandModel(x, y)
        >>> integrand.surrogate.plot()
        >>> coords = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> val, var = integrand.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        Example, 2d
        >>> x = np.random.rand(5, 2)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> integrand = IntegrandModel(x, y)
        >>> coords_x0 = np.linspace(0, 1, 10)
        >>> coords_x1 = np.linspace(0, 1, 10)
        >>> coords_meshgrid = np.meshgrid(coords_x0, coords_x1)
        >>> coords = np.hstack((coords_meshgrid[0].reshape(-1,1), coords_meshgrid[1].reshape(-1,1)))
        >>> val, var = integrand.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        """

        if prior is None:
            prior = np.ones(coords.shape[0]) / coords.shape[0]
        surrogate = self.surrogate.build_gp()
        self.quad_points = self.surrogate.x
        KxX = prior @ surrogate.kernel(coords, self.quad_points)
        KXX = (surrogate.kernel(self.quad_points, self.quad_points) +
               np.exp(self.surrogate.theta["log_jitter"]) * np.eye(len(self.quad_points), len(self.quad_points)))
        while np.linalg.det(KXX) < min_det:
            KXX += min_det * np.eye(len(self.quad_points), len(self.quad_points))
        self.quad_weights = np.linalg.solve(KXX, KxX)
        self.evidence = self.quad_weights @ self.surrogate.y

        # Down sample for calculating uncertainty efficiently
        reduction_interval = int(np.floor(np.sqrt(len(coords))).item())
        coords_reduced = coords[::reduction_interval]
        prior_reduced = prior[::reduction_interval]
        prior_reduced /= sum(prior_reduced)

        self.variance = sum(sum((surrogate.kernel(xi.reshape(1, -1), xj.reshape(1, -1))
                                - surrogate.kernel(xi.reshape(1, -1), self.quad_points)
                                @ np.linalg.solve(KXX, surrogate.kernel(self.quad_points, xj.reshape(1, -1)))) * Pxi
                                for xi, Pxi in zip(coords_reduced, prior_reduced)) * Pxj
                            for xj, Pxj in zip(coords_reduced, prior_reduced)).item()

        return self.evidence, self.variance

    def quad(self, n_samples: int = 100, sampler: callable = None, min_det: float = 0.001):
        """
        Calculates integral across unit hypercube, quadrature weights and uncertainty using Monte Carlo integration.
        :param n_samples: Number of points to sample. Defaults to 100
        :param sampler: Sampling method. Defaults to latin hypercube
        :param min_det: minimum determinant of quadrature point correlation matrix, to stabilise inversion
        :return: (Integral, Variance)
        Example, 1d:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> integrand = IntegrandModel(x, y)
        >>> integrand.surrogate.plot()
        >>> val, var = integrand.quad(1000)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> x = np.vstack((x, x))
        >>> y = np.sin(np.pi*x).reshape(-1) + 0.05 * np.random.randn(len(x))
        >>> integrand = IntegrandModel(x, y)
        >>> integrand.surrogate.plot()
        >>> val, var = integrand.quad(1000)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        Example, 2d
        >>> x = np.random.rand(5, 2)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> integrand = IntegrandModel(x, y)
        >>> val, var = integrand.quad(1000)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        """

        if sampler is None:
            sampler = qmc.LatinHypercube(d=self.surrogate.ndim).random

        samples = sampler(n_samples)

        return self.discrete_quad(samples, min_det=min_det)


class SqIntegrandModel(IntegrandModel):

    def __init__(self, x_init: np.ndarray,
                 y_init: np.ndarray,
                 kernel: tinygp.kernels.Kernel = tinygp.kernels.stationary.ExpSquared,
                 theta_init: dict = None,
                 optimize_init: bool = True):
        """
        Linearised square root warped Bayesian quadrature
        :param x_init: Input points used to initialise the surrogate GP, n_init X ndim
        :param y_init: Output point used to initialise the surrogate GP, n_init X 1
        :param kernel: Surrogate GP kernel, defaults to RBF Kernel
        :param theta_init: Initial values for kernel hyperparameters. Dict with fields "log_scale", "log_amp" and
        "log_jitter"
        :param optimize_init: If true, optimize hyperparameters on initial data. Forced to False if no initial data
        """

        super().__init__(x_init, y_init, kernel, theta_init, optimize_init, SqWarpedGP)

    def discrete_quad(self, coords: np.ndarray, prior: np.ndarray = None, min_det: float = 0.001):
        """
        Calculate summation (as an expectation) and quadrature weights for used quadrature points for given coords. Uses
        a constant mean function of the mean of the conditioning points. Also returns uncertainty in summation. To
        compute explicit sum not as an expectation use a prior of ones.
        :param coords: points to sum over
        :param prior: prior probabilities for each coord
        :param min_det: minimum determinant of quadrature point correlation matrix, to stabilise inversion
        :return: (summation, variance)
        Example, 1d:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> integrand = SqIntegrandModel(x, y)
        >>> integrand.surrogate.plot()
        >>> coords = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> val, var = integrand.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> x = np.vstack((x, x))
        >>> y = np.maximum(np.sin(np.pi*x).reshape(-1) + 0.05 * np.random.randn(len(x)), 0.1)
        >>> integrand = SqIntegrandModel(x, y)
        >>> integrand.surrogate.plot()
        >>> coords = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> val, var = integrand.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        Example, 2d
        >>> x = np.random.rand(5, 2)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> integrand = SqIntegrandModel(x, y)
        >>> coords_x0 = np.linspace(0, 1, 10)
        >>> coords_x1 = np.linspace(0, 1, 10)
        >>> coords_meshgrid = np.meshgrid(coords_x0, coords_x1)
        >>> coords = np.hstack((coords_meshgrid[0].reshape(-1,1), coords_meshgrid[1].reshape(-1,1)))
        >>> val, var = integrand.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {integrand.quad_points.reshape(-1)}, Weights: {integrand.quad_weights}')
        """

        if prior is None:
            prior = np.ones(coords.shape[0]) / coords.shape[0]
        surrogate = self.surrogate.build_gp()
        self.quad_points = self.surrogate.x
        KXxxX = surrogate.kernel(self.quad_points, coords) @ np.diag(prior) @ surrogate.kernel(coords, self.quad_points)
        KXX = (surrogate.kernel(self.quad_points, self.quad_points) +
               np.exp(self.surrogate.theta["log_jitter"]) * np.eye(len(self.quad_points), len(self.quad_points)))
        while np.linalg.det(KXX) < min_det:
            KXX += min_det * np.eye(len(self.quad_points), len(self.quad_points))
        KXX_i = np.linalg.inv(KXX)
        self.quad_weights = np.array(KXX_i @ KXxxX @ KXX_i)
        self.evidence = np.array(self.surrogate.epsilon + 0.5 * self.surrogate.y @ self.quad_weights @ self.surrogate.y)

        # Down sample for calculating uncertainty efficiently
        reduction_interval = int(np.floor(np.sqrt(len(coords))).item())
        coords_reduced = coords[::reduction_interval]
        prior_reduced = prior[::reduction_interval]
        prior_reduced /= sum(prior_reduced)

        KXxxX = (surrogate.kernel(self.quad_points, coords_reduced) @ np.diag(prior_reduced)
                 @ surrogate.kernel(coords_reduced, self.quad_points))
        KxyxXyX = sum(sum((surrogate.kernel(xi.reshape(1, -1), xj.reshape(1, -1))
                           * surrogate.kernel(self.quad_points, xi.reshape(1, -1))
                           @ surrogate.kernel(xj.reshape(1, -1), self.quad_points) * Pxi
                           for xi, Pxi in zip(coords_reduced, prior_reduced))) * Pxj
                      for xj, Pxj in zip(coords_reduced, prior_reduced))
        variance_weights = KXX_i @ (KxyxXyX - KXxxX @ KXX_i @ KXxxX) @ KXX_i
        self.variance = np.array(self.surrogate.y @ variance_weights @ self.surrogate.y)

        return self.evidence, self.variance
