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
        self.discrete_quad_calculated = False
        self.quad_weights = None
        self.discrete_quad_weights = None
        self.discrete_quad_uncertainty = None

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

    def optimize_theta(self, La: float = 0.1, Ls: float = 0.01):
        """
        Optimize hyperparameters over current data
        :param La: weighting of regularisation term on log amplitude
        :param Ls: weighting of regularisation term on scale
        """
        func = lambda theta: (self.negative_log_likelihood(theta) + La * theta["log_amp"] ** 2
                              + Ls * theta["log_scale"] @ theta["log_scale"])
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
        self.discrete_quad_calculated = False
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
        >>> x = QuadGP.get_init_points(5,1, lbs, ubs)
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
        >>> x = QuadGP.get_init_points(20,2, lbs, ubs)
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
        >>> x = QuadGP.get_init_points(5,1, lbs, ubs)
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


    def numerical_quad(self, lbs: list, ubs: list):
        """
        Calculate integral and quadrature weights for used quadrature points using numerical integration. Deprecated
        for RBF kernels, instead use quad.
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
        >>> print(quad_gp.numerical_quad(lbs, ubs))
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 2d
        >>> lbs = [-5, -5]
        >>> ubs = [5, 5]
        >>> x = QuadGP.get_init_points(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> print(quad_gp.numerical_quad(lbs, ubs))
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

    def quad(self, lbs: list, ubs: list):
        """
        Calculate integral and quadrature weights for used quadrature points exactly for RBF kernels.
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
        Example, 1d with acquisition function:
        >>> lbs = -5
        >>> ubs = 5
        >>> x = QuadGP.get_init_points(1, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> for i in range(10):
                next_x = quad_gp.acquire_next(lbs + np.random.rand() * (ubs - lbs), lbs, ubs)
                next_y = np.exp(-next_x**2)
                quad_gp.add_data(next_x, next_y)
                quad_gp.plot_gp(lbs, ubs)
                val= quad_gp.quad(lbs, ubs)
                print(f'Integral value: {val}')
                print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        Example, 2d
        >>> lbs = [-5, -5]
        >>> ubs = [5, 5]
        >>> x = QuadGP.get_init_points(10, 2, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y)
        >>> print(quad_gp.quad(lbs, ubs))
        >>> print(f'Quadrature points: {quad_gp.x}, Weights: {quad_gp.quad_weights}')
1        """
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
            KXX = surrogate.kernel(self.x, self.x)
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
        Calculate summation and quadrature weights for used quadrature points for given coords. Uses a constant mean
        function of the mean of the conditioning points. Also returns uncertainty in summation.
        :param coords: points to sum over
        :param prior: prior probabilities for each coord
        :return: (summation results, summation variance)
        Example, 1d:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.get_init_points(5, 1, lbs, ubs)
        >>> y = np.exp(-(x**2).sum(axis=1))
        >>> quad_gp = QuadGP(x, y, mean='avg')
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> coords = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        >>> val, var = quad_gp.discrete_quad(coords)
        >>> print(f'Summation value: {val}, Uncertainty: {var}')
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.discrete_quad_weights}')
        Example, 1d with acquisition function:
        >>> lbs = -1
        >>> ubs = 1
        >>> x = QuadGP.get_init_points(1, 1, lbs, ubs)
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
        """
        if self.discrete_quad_calculated:
            return self.discrete_quad_weights @ self.y, self.discrete_quad_uncertainty
        else:
            if prior is None:
                prior = jnp.ones(coords.shape[0]) / coords.shape[0]
            surrogate = self.build_gp(self.theta)
            KxX = prior @ surrogate.kernel(coords, self.x)
            KXX = surrogate.kernel(self.x, self.x)
            zero_mean_weights = jnp.linalg.solve(KXX, KxX)
            if self.update_mean:
                nX = self.x.shape[0]
                self.discrete_quad_weights = (jnp.ones(nX) / nX + (jnp.eye(nX) - jnp.ones((nX, nX)) / nX)
                                              @ zero_mean_weights)
            else:
                self.discrete_quad_weights = zero_mean_weights

            self.discrete_quad_uncertainty = sum((surrogate.kernel(xi.reshape(-1,1), xi.reshape(-1,1)
                                                  - surrogate.kernel(xi.reshape(-1,1), self.x)
                                                  @ jnp.linalg.solve(KXX, surrogate.kernel(self.x,
                                                                                           xi.reshape(-1,1)))) * Pxi
                                                 for xi, Pxi in zip(coords, prior))).item()
            self.discrete_quad_calculated = True
            return self.discrete_quad_weights @ self.y, self.discrete_quad_uncertainty

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
