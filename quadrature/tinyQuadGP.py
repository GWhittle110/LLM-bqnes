# Import required packages
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
import numpy as np
import tinygp
import scipy.integrate


class QuadGP:

    def __init__(self, x_init: jnp.ndarray,
                 y_init: jnp.ndarray,
                 ndim: int = None,
                 theta_init: dict = None,
                 optimize_init: bool = True,
                 mean=0):
        """
        Gaussian process to carry out Bayesian quadrature using exact inference
        :param x_init: Input points used to initialise the GP, n_init X ndim
        :param y_init: Output point used to initialise the GP, n_init X 1
        :param ndim: Dimensionality of input data. Redundant if x_init provided
        :param theta_init: Initial values for kernel hyperparameters. Dict with fields "log_scale", "log_amp" and
        "log_jitter"
        :param optimize_init: If true, optimize hyperparameters on initial data. Forced to False if no initial data
        :param mean: Method for computing mean or constant value for mean. Currently only 0 mean and 'avg' implemented
        for quadrature routines
        Example:
        >>> x_init = QuadGP.latin_hypercube(5, 1)
        >>> y_init = x_init**2
        >>> GP = QuadGP(x_init, y_init)
        """

        if x_init is None:
            self.x = jnp.empty(ndim)
            self.y = jnp.empty(1)
            self.ndim = ndim
        else:
            self.x = x_init
            self.y = y_init
            self.ndim = x_init.shape[1]

        if mean == 'avg':
            self.update_mean = True
            self.mean = tinygp.means.Mean(jnp.mean(y_init))
        else:
            self.update_mean = False
            self.mean = tinygp.means.Mean(mean)

        if theta_init is None:
            self.theta = {"log_scale": -1 * np.ones(self.ndim),
                          "log_amp": np.float64(0),
                          "log_jitter": np.float64(0)}
        else:
            self.theta = theta_init

        if optimize_init and x_init is not None:
            self.optimize_theta()

        self.quad_calculated = False
        self.quad_weights = None
        self.quad_uncertainty = None

    def get_kernel(self, theta: dict = None):
        """
        Generate GP kernel from hyperparameters. By default, uses RBF kernel
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: tinygp.kernel
        """
        if theta is None:
            theta = self.theta
        return jnp.exp(theta["log_amp"]) * tinygp.transforms.Linear(jnp.exp(-theta["log_scale"]),
                                                                     tinygp.kernels.ExpSquared())

    def build_gp(self, theta: dict = None):
        """
        Instantiate the gaussian process object
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: tinygp.GaussianProcess
        """
        if theta is None:
            theta = self.theta
        return tinygp.GaussianProcess(mean=self.mean, kernel=self.get_kernel(theta), X=self.x, diag=jnp.exp(theta["log_jitter"]))

    def negative_log_likelihood(self, theta: dict = None):
        """
        Calculate negative log likelihood of y values conditioned on the GP induced by x and the kernel
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: float, value of negative log likelihood
        """
        if theta is None:
            theta = self.theta
        return -self.build_gp(theta).log_probability(self.y)

    def optimize_theta(self, La: float = 0.1, Ls: float = 0.1):
        """
        Optimize hyperparameters over current data
        :param La: weighting of regularisation term on log amplitude
        :param Ls: weighting of regularisation term on scale
        """
        func = lambda theta: (self.negative_log_likelihood(theta) + La * theta["log_amp"] ** 2
                              + Ls * jnp.exp(theta["log_scale"]) @ jnp.exp(theta["log_scale"]))
        minimizer = jaxopt.ScipyMinimize(fun=func)
        self.theta = minimizer.run(init_params=self.theta).params

    def add_data(self, x, y):
        """
        Add data to gp
        :param x: np.ndarray, evaluation point
        :param y: np.ndarray, evaluation value
        """
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        if self.update_mean:
            self.mean = tinygp.means.Mean(jnp.mean(self.y))
        self.quad_calculated = False
        self.optimize_theta()

    def acquire_next(self, x_init: np.ndarray, lbs: list = None, ubs: list = None, method: str = "Variance"):
        """
        Optimize acquisition function to determine next point to sample
        :param x_init: Point to initialise minimization routine from
        :param lbs: lower bounds on next point
        :param ubs: upper bounds on next point
        :param method: str, objective function to be maximised. Defaults to variance
        :return: torch.Tensor, next sample point
        Example, 1d:
        >>> lbs = -5
        >>> ubs = 5
        >>> x = QuadGP.latin_hypercube(5,1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> x_next = quad_gp.acquire_next(np.ones(1), lbs, ubs)
        >>> y_next = np.exp(-(x_next**2).sum())
        >>> print(f'Acquired sample: {x_next}, corresponding function evaluation: {y_next}')
        >>> quad_gp.add_data(x_next, y_next)
        >>> quad_gp.plot_gp(lbs, ubs)
        Example, 2d:
        >>> lbs = [-5, -5]
        >>> ubs = [5, 5]
        >>> x = QuadGP.latin_hypercube(20,2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> x_next = quad_gp.acquire_next(np.ones(2), lbs, ubs)
        >>> y_next = np.exp(-(x_next**2).sum())
        >>> print(f'Acquired sample: {x_next}, corresponding function evaluation: {y_next}')
        """
        surrogate = self.build_gp()
        if method == "Variance":
            objective = lambda x: -1 * jnp.sqrt(surrogate.condition(self.y,
                                                                    X_test=x.reshape((-1, self.ndim))).gp.variance)[0]
        else:
            raise NotImplementedError
        minimizer = jaxopt.ScipyMinimize(fun=objective)
        next_x = minimizer.run(x_init).params
        if lbs is not None:
            next_x = jnp.maximum(next_x, jnp.array(lbs))
        if ubs is not None:
            next_x = jnp.minimum(next_x, jnp.array(ubs))

        return next_x

    def acquire_next_discrete(self, points: np.ndarray, method: str = "Variance"):
        """
        Optimize acquisition function to determine next point to sample from a discrete set of points
        :param points: Candidate points
        :param method: str, objective function to be maximised. Defaults to variance
        :return: torch.Tensor, next sample point
        Example, 1d:
        >>> lbs = -5
        >>> ubs = 5
        >>> x = QuadGP.latin_hypercube(5,1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> points = np.linspace(lbs, ubs).reshape(-1, 1)
        >>> x_next = quad_gp.acquire_next_discrete(points)
        >>> y_next = np.exp(-(x_next**2).sum())
        >>> print(f'Acquired sample: {x_next}, corresponding function evaluation: {y_next}')
        >>> quad_gp.add_data(x_next, y_next)
        >>> quad_gp.plot_gp(lbs, ubs)
        """
        surrogate = self.build_gp()
        if method == "Variance":
            objective = lambda x: surrogate.condition(self.y, X_test=x.reshape((-1, self.ndim))).gp.variance
        else:
            raise NotImplementedError
        next_x = points[jnp.argmax(objective(points))]
        return next_x


    def scipy_quad(self, lbs: list, ubs: list):
        """
        Calculate integral and quadrature weights for used quadrature points using numerical integration. Deprecated
        for RBF kernels, instead use quad.
        :param lbs: lower bounds for integration
        :param ubs: upper bounds for integration
        :return: integral result
        Example, 1d:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> coords = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        >>> val = quad_gp.scipy_quad(lbs, ubs)
        >>> print(f'Integral value: {val}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> x = np.vstack((x, x))
        >>> y = np.exp(-(x**2).sum(axis=1)) + 0.05 * np.random.randn(len(x))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> val = quad_gp.scipy_quad(lbs, ubs)
        >>> print(f'Integral value: {val}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 1d with acquisition function:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(1, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand() * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-next_x**2)
                quad_gp.add_data(next_x, next_y)
                quad_gp.plot_gp(lbs, ubs)
                val= quad_gp.scipy_quad(lbs, ubs)
                print(f'Integral value: {val}')
                print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')

        Example, 2d
        >>> lbs = [-1, -1]
        >>> ubs = [1, 1]
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> val = quad_gp.scipy_quad(lbs, ubs)
        >>> print(f'Integral value: {val}')
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        Example, 2d with acquisition function:
        >>> lbs = np.array([-1, -1])
        >>> ubs = np.array([1, 1])
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand(2) * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-(next_x**2).sum())
                quad_gp.add_data(next_x, next_y)
                val= quad_gp.scipy_quad(lbs, ubs)
                print(f'Integral value: {val}')
                print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        """
        if self.quad_calculated:
            return self.quad_weights @ self.y
        else:
            surrogate = self.build_gp()
            if self.ndim == 1:
                KxX = jnp.array([scipy.integrate.quad(lambda x: surrogate.kernel(np.array(x).reshape((1, -1)),
                                                                                 X.reshape((1, -1))),
                                                      lbs, ubs)[0] for X in self.x])
            else:
                KxX = jnp.array([scipy.integrate.nquad(lambda *x: surrogate.kernel(np.array(x).reshape((1, -1)),
                                                                                   X.reshape((1, -1))),
                                                       [*zip(lbs, ubs)])[0] for X in self.x])
            KXX = (surrogate.kernel(self.x, self.x) +
                   jnp.exp(self.theta["log_jitter"]) * jnp.eye(len(self.x), len(self.x)))
            self.quad_weights = jnp.linalg.solve(KXX, KxX)
            self.quad_calculated = True
            return self.quad_weights @ self.y

    def rbf_quad(self, lbs: list, ubs: list):
        """
        Calculate integral and quadrature weights for used quadrature points exactly for RBF kernels.
        :param lbs: lower bounds for integration
        :param ubs: upper bounds for integration
        :return: integral result
        Example, 1d:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> coords = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        >>> val = quad_gp.rbf_quad(lbs, ubs)
        >>> print(f'Integral value: {val}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> x = np.vstack((x, x))
        >>> y = np.exp(-(x**2).sum(axis=1)) + 0.05 * np.random.randn(len(x))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> val = quad_gp.rbf_quad(lbs, ubs)
        >>> print(f'Integral value: {val}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 1d with acquisition function:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(1, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand() * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-next_x**2)
                quad_gp.add_data(next_x, next_y)
                quad_gp.plot_gp(lbs, ubs)
                val= quad_gp.rbf_quad(lbs, ubs)
                print(f'Integral value: {val}')
                print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 2d
        >>> lbs = [-1, -1]
        >>> ubs = [1, 1]
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> val = quad_gp.rbf_quad(lbs, ubs)
        >>> print(f'Integral value: {val}')
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        Example, 2d with acquisition function:
        >>> lbs = np.array([-1, -1])
        >>> ubs = np.array([1, 1])
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand(2) * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-(next_x**2).sum())
                quad_gp.add_data(next_x, next_y)
                val= quad_gp.rbf_quad(lbs, ubs)
                print(f'Integral value: {val}')
                print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        """
        if self.quad_calculated:
            return self.quad_weights @ self.y
        else:
            surrogate = self.build_gp()
            sig = jnp.exp(self.theta["log_scale"])
            if self.ndim == 1:
                if isinstance(lbs, list):
                    lbs = lbs[0]
                if isinstance(ubs, list):
                    ubs = ubs[0]
                vol = ubs-lbs
                KxX = (jnp.exp(self.theta["log_amp"]) * sig * jnp.sqrt(2 * jnp.pi) *
                       jnp.array([norm.cdf((ubs - x) / sig)-norm.cdf((lbs - x) / sig) for x in self.x]).reshape(-1))
            else:
                vol = jnp.prod(jnp.array([ub-lb for ub, lb in zip(ubs, lbs)]))
                KxX = (jnp.exp(self.theta["log_amp"]) * jnp.sqrt(2 * jnp.pi)**self.ndim *
                       jnp.array([jnp.prod(jnp.array([sig[i]*norm.cdf((ub-x[i])/sig[i])-norm.cdf((lb-x[i])/sig[i])
                                                      for i, (ub, lb) in enumerate(zip(ubs, lbs))]).reshape(-1))
                                  for x in self.x]).reshape(-1))
            KXX = (surrogate.kernel(self.x, self.x) +
                   jnp.exp(self.theta["log_jitter"]) * jnp.eye(len(self.x), len(self.x)))
            zero_mean_weights = jnp.linalg.solve(KXX, KxX)
            if self.update_mean:
                nX = self.x.shape[0]
                self.quad_weights = (jnp.ones(nX) * vol / nX + (jnp.eye(nX) - jnp.ones((nX, nX)) / nX)
                                              @ zero_mean_weights)
            else:
                self.quad_weights = zero_mean_weights
            self.quad_calculated = True
            return self.quad_weights @ self.y

    def discrete_quad(self, coords: np.ndarray, prior: np.ndarray = None):
        """
        Calculate summation (as an expectation) and quadrature weights for used quadrature points for given coords. Uses
        a constant mean function of the mean of the conditioning points. Also returns uncertainty in summation. To
        compute explicit sum not as an expectation use a prior of ones.
        :param coords: points to sum over
        :param prior: prior probabilities for each coord
        :return: (summation results, summation variance)
        Example, 1d:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> coords = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        >>> val, var = quad_gp.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Uncertainty: {var}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> x = np.vstack((x, x))
        >>> y = np.exp(-(x**2).sum(axis=1)) + 0.05 * np.random.randn(len(x))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> coords = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        >>> val, var = quad_gp.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Uncertainty: {var}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 1d with acquisition function:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(1, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> coords = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next_discrete(coords)
                next_y = np.exp(-next_x**2)
                quad_gp.add_data(next_x, next_y)
                quad_gp.plot_gp(lbs, ubs)
                val, var = quad_gp.discrete_quad(coords)
                print(f'Summation value: {val}, Uncertainty: {var}')
                print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.discrete_quad_weights}')
        Example, 2d
        >>> lbs = [-1, -1]
        >>> ubs = [1, 1]
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> coords_x0 = np.linspace(lbs[0], ubs[0], 10)
        >>> coords_x1 = np.linspace(lbs[1], ubs[1], 10)
        >>> coords_meshgrid = np.meshgrid(coords_x0, coords_x1)
        >>> coords = np.hstack((coords_meshgrid[0].reshape(-1,1), coords_meshgrid[1].reshape(-1,1)))
        >>> val, var = quad_gp.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Variance: {var}')
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        Example, 2d with acquisition function:
        >>> lbs = np.array([-1, -1])
        >>> ubs = np.array([1, 1])
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand(2) * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-(next_x**2).sum())
                quad_gp.add_data(next_x, next_y)
                coords_x0 = np.linspace(lbs[0], ubs[0], 10)
                coords_x1 = np.linspace(lbs[1], ubs[1], 10)
                coords_meshgrid = np.meshgrid(coords_x0, coords_x1)
                coords = np.hstack((coords_meshgrid[0].reshape(-1,1), coords_meshgrid[1].reshape(-1,1)))
                val, var = quad_gp.discrete_quad(coords)
                print(f'Summation value: {val}, Variance: {var}')
                print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.discrete_quad_weights}')
        """
        if self.quad_calculated:
            return self.quad_weights @ self.y, self.quad_uncertainty
        else:
            if prior is None:
                prior = jnp.ones(coords.shape[0]) / coords.shape[0]
            surrogate = self.build_gp(self.theta)
            KxX = prior @ surrogate.kernel(coords, self.x)
            KXX = (surrogate.kernel(self.x, self.x) +
                   jnp.exp(self.theta["log_jitter"]) * jnp.eye(len(self.x), len(self.x)))
            zero_mean_weights = jnp.linalg.solve(KXX, KxX)
            if self.update_mean:
                nX = self.x.shape[0]
                self.quad_weights = (jnp.ones(nX) / nX + (jnp.eye(nX) - jnp.ones((nX, nX)) / nX)
                                     @ zero_mean_weights) * jnp.sum(prior)
            else:
                self.quad_weights = zero_mean_weights

            # Down sample for calculating uncertainty efficiently
            reduction_interval = int(np.floor(np.sqrt(len(coords))).item())
            coords_reduced = coords[::reduction_interval]
            prior_reduced = prior[::reduction_interval]
            prior_reduced /= sum(prior_reduced)

            self.quad_uncertainty = sum(sum((surrogate.kernel(xi.reshape(1, -1), xj.reshape(1, -1))
                                            - surrogate.kernel(xi.reshape(1, -1), self.x)
                                            @ jnp.linalg.solve(KXX,
                                                               surrogate.kernel(self.x, xj.reshape(1, -1)))) * Pxi
                                            for xi, Pxi in zip(coords_reduced, prior_reduced)) * Pxj
                                        for xj, Pxj in zip(coords_reduced, prior_reduced)).item()
            self.quad_calculated = True
            return self.quad_weights @ self.y, self.quad_uncertainty

    def mc_quad(self, n_samples: int = 100, lbs: list = None, ubs: list = None, sampler: callable = None,
                explicit: bool = False):
        """
        Calculates integral (as an expectation), quadrature weights and uncertainty using Monte Carlo integration. To
        compute the explicit integral (not as an expectation) use a uniform sampler and set explicit to True.
        :param n_samples: Number of points to sample. Defaults to 100
        :param lbs: Integral lower bounds. Defaults to 0
        :param ubs: Integral upper bounds. Defaults to 1
        :param sampler: Sampling method. Defaults to latin hypercube
        :param explicit: Whether to compute the integral explicitly instead of as an expectation.
        :return: (Integral results, Integral variance)
        Example, 1d:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(10, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> val, var = quad_gp.mc_quad(100, lbs, ubs, explicit=True)
        >>> print(f'Integral value: {val}, Integral variance: {var}')
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        Example, 1d with repeated measurements and noise:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(5, 1, lbs, ubs)
        >>> x = np.vstack((x, x))
        >>> y = np.exp(-(x**2).sum(axis=1)) + 0.05 * np.random.randn(len(x))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> val, var = quad_gp.mc_quad(100, lbs, ubs, explicit=True)
        >>> print(f'Integral value: {val}, Integral variance: {var}')
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        Example, 1d with acquisition function:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.latin_hypercube(1, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand() * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-next_x**2)
                quad_gp.add_data(next_x, next_y)
                quad_gp.plot_gp(lbs, ubs)
                val, var = quad_gp.mc_quad(100, lbs, ubs, explicit=True)
                print(f'Integral value: {val}, Integral variance: {var}')
                print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')

        Example, 2d
        >>> lbs = [-1, -1]
        >>> ubs = [1, 1]
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> val, var = quad_gp.mc_quad(100, lbs, ubs, explicit=True)
        >>> print(f'Integral value: {val}, Integral variance: {var}')
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        Example, 2d with acquisition function:
        >>> lbs = np.array([-1, -1])
        >>> ubs = np.array([1, 1])
        >>> x = QuadGP.latin_hypercube(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand(2) * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-(next_x**2).sum())
                quad_gp.add_data(next_x, next_y)
                val, var = quad_gp.mc_quad(100, lbs, ubs, explicit=True)
                print(f'Integral value: {val}, Integral variance: {var}')
                print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
        """

        if lbs is None:
            lbs = [0] * self.ndim
        if ubs is None:
            ubs = [1] * self.ndim
        if sampler is None:
            sampler = self.latin_hypercube

        samples = sampler(n_points=n_samples, ndim=self.ndim, lbs=lbs, ubs=ubs)

        if explicit:
            try:
                prior = jnp.ones(n_samples) / n_samples * jnp.prod(jnp.array([ub - lb
                                                                              for lb, ub in zip(ubs, lbs)])).item()
            except TypeError:
                prior = jnp.ones(n_samples) / n_samples * (ubs-lbs)
        else:
            prior = None

        return self.discrete_quad(samples, prior)


    def wsabi_quad(self, epsilon, n_samples: int = 100, lbs: list = None, ubs: list = None, sampler: callable = None):
        """
        Calculates integral (as an expectation), quadrature weights and uncertainty using Monte Carlo integration,
        enforcing a positive integrand using the wsabi-L warping method.
        :param n_samples: Number of points to sample. Defaults to 100
        :param lbs: Integral lower bounds. Defaults to 0
        :param ubs: Integral upper bounds. Defaults to 0
        :param sampler: Sampling method. Defaults to latin hypercube
        :return: (Integral results, Integral variance)
        >>> x = QuadGP.latin_hypercube(20, 1, 0, 1)
        >>> y = (np.sin(10*x)**2).reshape(-1) + 0.2
        >>> epsilon = 0.8 * np.min(y)
        >>> z = np.sqrt(2*(y-epsilon))
        >>> quad_gp = QuadGP(x, z)
        >>> weights = quad_gp.wsabi_quad(epsilon)
        >>> integral_weights = np.diag(weights) / np.trace(weights)
        >>> print(y @ integral_weights)
        """

        if lbs is None:
            lbs = [0] * self.ndim
        if ubs is None:
            ubs = [1] * self.ndim
        if sampler is None:
            sampler = self.latin_hypercube

        samples = sampler(n_points=n_samples, ndim=self.ndim, lbs=lbs, ubs=ubs)
        prior = jnp.ones(samples.shape[0]) / samples.shape[0]

        surrogate = self.build_gp(self.theta)

        KXxKxX = (surrogate.kernel(self.x, samples) @ np.diag(prior) @ surrogate.kernel(samples, self.x))
        KXX = (surrogate.kernel(self.x, self.x) +
               (epsilon + np.exp(self.theta["log_jitter"])) * jnp.eye(len(self.x), len(self.x)))
        KXX_i = np.linalg.inv(KXX)

        weights = KXX_i @ KXxKxX @ KXX_i
        print(weights)
        integral = epsilon + 0.5 * self.y @ weights @ self.y
        print(integral)
        return weights




    def plot_gp(self, lbs, ubs):
        """
        Plot 1d Gaussian process
        :return: None
        """
        test_x = np.linspace(lbs, ubs, 100).reshape((-1, 1))
        surrogate = self.build_gp()
        test_y = surrogate.condition(self.y, X_test=test_x).gp.loc
        test_sd = np.sqrt(surrogate.condition(self.y, X_test=test_x).gp.variance)


        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(self.x, self.y, 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.flatten(), test_y.flatten(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.flatten(), test_y.flatten() - 2 * test_sd, test_y + 2 * test_sd, alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()

    @staticmethod
    def latin_hypercube(n_points: int, ndim: int, lbs=None, ubs=None):
        """
        Helper function to sample points using Latin hypercube
        :param n_points: Number of points to sample
        :param ndim: Number of dimensions to sample in
        :param lbs: Lower bounds for rescaling points
        :param ubs: Upper bounds for rescaling points
        :return: Array of initialisation points
        Example:
        >>> n_points = 5
        >>> ndim = 3
        >>> lbs = [-1, -1, -1]
        >>> ubs = [1, 1, 1]
        >>> samples = QuadGP.latin_hypercube(n_points, ndim, lbs, ubs)
        >>> print(samples)
        """
        if lbs is None:
            lbs = [0] * ndim
        if ubs is None:
            ubs = [1] * ndim
        sampler = qmc.LatinHypercube(d=ndim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, lbs, ubs)
