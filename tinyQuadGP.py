# Import required packages
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import tinygp
import scipy.integrate


class QuadGP:

    def __init__(self, x_init: jnp.ndarray,
                 y_init: jnp.ndarray,
                 ndim: int = None,
                 theta_init: dict = None,
                 optimize_init: bool = True):
        """
        Gaussian process to carry out Bayesian quadrature using exact inference
        :param x_init: Input points used to initialise the GP, n_init X ndim
        :param y_init: Output point used to initialise the GP, n_init X 1
        :param ndim: Dimensionality of input data. Redundant if x_init provided
        :param theta_init: Initial values for kernel hyperparameters. Dict with fields "log_scale", "log_amp" and
        "log_jitter"
        :param optimize_init: If true, optimize hyperparameters on initial data. Forced to False if no initial data
        Example:
        >>> x_init = QuadGP.get_init_points(5, 1)
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

        if theta_init is None:
            self.theta = {"log_scale": np.float64(0),
                          "log_amp": np.float64(0),
                          "log_jitter": np.float64(0.01)}
        else:
            self.theta = theta_init

        if optimize_init and x_init is not None:
            self.optimize_theta()

        self.quad_calculated = False
        self.quad_weights = None

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
        return tinygp.GaussianProcess(kernel=self.get_kernel(theta), X=self.x, diag=jnp.exp(theta["log_jitter"]))

    def negative_log_likelihood(self, theta: dict = None):
        """
        Calculate negative log likelihood of y values conditioned on the GP induced by x and the kernel
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: float, value of negative log likelihood
        """
        if theta is None:
            theta = self.theta
        return -self.build_gp(theta).log_probability(self.y)

    def optimize_theta(self):
        """
        Optimize hyperparameters over current data
        """
        minimizer = jaxopt.ScipyMinimize(fun=self.negative_log_likelihood)
        self.theta = minimizer.run(init_params=self.theta).params

    def add_data(self, x, y):
        """
        Add data to gp
        :param x: np.ndarray, evaluation point
        :param y: np.ndarray, evaluation value
        """
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))

    def acquire_next(self, x_init: np.ndarray, method: str = "Variance"):
        """
        Optimize acquisition function to determine next point to sample
        :param x_init: Point to initialise minimization routine from
        :param method: str, objective function to be maximised. Defaults to variance
        :return: torch.Tensor, next sample point
        Example, 1d:
        >>> lbs = -5
        >>> ubs = 5
        >>> x = QuadGP.get_init_points(5,1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> x_next = quad_gp.acquire_next(np.ones(2))
        >>> y_next = np.exp(-(x_next**2).sum())
        >>> print(f'Acquired sample: {x_next}, corresponding function evaluation: {y_next}')
        >>> quad_gp.add_data(x_next, y_next)
        >>> quad_gp.plot_gp(lbs, ubs)
        Example, 2d:
        >>> x = QuadGP.get_init_points(20,2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> x_next = quad_gp.acquire_next(np.ones(2))
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
        return next_x

    def quad(self, lbs: list, ubs: list):
        """
        Calculate integral and quadrature weights for used quadrature points
        :param lbs: lower bounds for integration
        :param ubs: upper bounds for integration
        :return: integral result
        Example, 1d:
        >>> lbs = -5
        >>> ubs = 5
        >>> x = QuadGP.get_init_points(10, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> print(quad_gp.quad(lbs, ubs))
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 2d
        >>> lbs = [-5, -5]
        >>> ubs = [5, 5]
        >>> x = QuadGP.get_init_points(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> print(quad_gp.quad(lbs, ubs))
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
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
            KXX = surrogate.kernel(self.x, self.x)
            self.quad_weights = jnp.linalg.solve(KXX, KxX)
            self.quad_calculated = True
            return self.quad_weights @ self.y

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
    def get_init_points(n_points: int, ndim: int, lbs=None, ubs=None):
        """
        Helper function to generate initialisation points using Latin hypercube
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
        >>> samples = QuadGP.get_init_points(n_points, ndim, lbs, ubs)
        >>> print(samples)
        """
        if lbs is None:
            lbs = [0] * ndim
        if ubs is None:
            ubs = [1] * ndim
        sampler = qmc.LatinHypercube(d=ndim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, lbs, ubs)
