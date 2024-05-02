"""
Gaussian process surrogate model for integrand
"""

import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import tinygp


class GP:

    def __init__(self, x_init: jnp.ndarray,
                 y_init: jnp.ndarray,
                 kernel: type = tinygp.kernels.stationary.ExpSquared,
                 theta_init: dict = None,
                 theta_anisotropic: bool = True,
                 optimize_init: bool = True):
        """
        Gaussian process model
        :param x_init: Input points used to initialise the GP, n_init X ndim
        :param y_init: Output point used to initialise the GP, n_init X 1
        :param kernel: GP kernel
        :param theta_init: Initial values for kernel hyperparameters. Dict with fields "log_scale", "log_amp" and
        "log_jitter"
        :param theta_anisotropic: Whether to use anisotropic length scale for kernel
        :param optimize_init: If true, optimize hyperparameters on initial data. Forced to False if no initial data
        """

        self.x = x_init
        self.y = y_init
        self.ndim = x_init.shape[1]

        self.kernel = kernel

        self.default_theta = {"log_scale": np.float64(-1),
                              "log_amp": np.float64(0),
                              "log_jitter": np.float64(0)}

        if theta_init is None:
            self.theta = self.default_theta
        else:
            self.theta = theta_init

        if theta_anisotropic and len(self.theta["log_scale"].shape) == 0:
            self.theta["log_scale"] *= np.ones(self.ndim)

        self.default_theta = self.theta

        if optimize_init and x_init is not None:
            self.optimize_theta()

    def get_kernel(self, theta: dict = None):
        """
        Generate GP kernel from hyperparameters
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: tinygp.kernel
        """
        if theta is None:
            theta = self.theta
        return jnp.exp(theta["log_amp"]) * tinygp.transforms.Linear(jnp.exp(-theta["log_scale"]), self.kernel())

    def build_gp(self, theta: dict = None):
        """
        Instantiate the gaussian process object
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: tinygp.GaussianProcess
        """
        if theta is None:
            theta = self.theta
        return tinygp.GaussianProcess(kernel=self.get_kernel(theta=theta), X=self.x, diag=jnp.exp(theta["log_jitter"]))

    def negative_log_likelihood(self, theta: dict = None):
        """
        Calculate negative log likelihood of y values conditioned on the GP induced by x and the kernel
        :param theta: hyperparameters, specified separately for optimisation purposes
        :return: float, value of negative log likelihood
        """
        if theta is None:
            theta = self.theta
        return -self.build_gp(theta).log_probability(self.y)

    def optimize_theta(self, La: float = 0.0001, Ls: float = 0.0001, Lj: float = 0.0001):
        """
        Optimize hyperparameters over current data
        :param La: weighting of regularisation term on log amplitude (keeps amplitude close to 1)
        :param Ls: weighting of regularisation term on scale
        :param Lj: weighting of regularisation term on log jitter (keeps jitter close to 1)
        """
        def fun(theta):
            return (self.negative_log_likelihood(theta) + La * theta["log_amp"] ** 2 + Lj * theta["log_jitter"] ** 2 +
                    Ls * jnp.exp(theta["log_scale"]) @ jnp.exp(theta["log_scale"]))
        minimizer = jaxopt.ScipyMinimize(fun=fun)
        self.theta = minimizer.run(init_params=self.default_theta).params

    def add_data(self, x, y, optimize_theta=True, *args, **kwargs):
        """
        Add data to gp
        :param x: np.ndarray, evaluation point
        :param y: np.ndarray, evaluation value
        :param optimize_theta: Whether to re-optimize hyperparameters
        """
        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        if optimize_theta:
            self.optimize_theta(*args, **kwargs)

    def predict(self, x):
        """
        Calculate mean and variance of predicted value at queried x point
        :param x: query point
        :return: mean, variance
        """
        gp = self.build_gp().condition(self.y, X_test=x, diag=jnp.exp(self.theta["log_jitter"])).gp
        return gp.loc, gp.variance

    def plot(self):
        """
        Plot 1d Gaussian process
        :return: None
        """
        lbs = 0
        ubs = 1
        test_x = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        test_y, variance = self.predict(test_x)
        test_sd = jnp.sqrt(variance)


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


class SqWarpedGP(GP):

    def __init__(self, x_init: jnp.ndarray = None,
                 y_init: jnp.ndarray = None,
                 kernel: type = tinygp.kernels.stationary.ExpSquared,
                 theta_init: dict = None,
                 theta_anisotropic: bool = True,
                 optimize_init: bool = True,
                 parent_gp: GP = None):
        """
        Square root warped Gaussian process model
        :param x_init: Input points used to initialise the GP, n_init X ndim
        :param y_init: Output point used to initialise the GP, n_init X 1
        :param kernel: GP kernel
        :param theta_init: Initial values for kernel hyperparameters. Dict with fields "log_scale", "log_amp" and
        "log_jitter"
        :param theta_anisotropic: Whether to use anisotropic length scale for kernel
        :param optimize_init: If true, optimize hyperparameters on initial data. Forced to False if no initial data
        :param parent_gp: Use in place of x_init and y_init arguments. Takes a standard GP and converts it to a square
        warped GP
        Example, GP conversion:
        >>> x = np.random.rand(5).reshape(-1,1)
        >>> y = np.sin(np.pi*x).reshape(-1)
        >>> parent_gp = GP(x, y)
        >>> parent_gp.plot()
        >>> child_gp = SqWarpedGP(parent_gp=parent_gp, optimize_init=False)
        >>> child_gp.plot()
        """

        if parent_gp is not None:
            x_init = parent_gp.x
            y_init = parent_gp.y
            theta_init = parent_gp.theta

        self.y_unwarped = y_init
        self.epsilon = 0.8 * np.min(self.y_unwarped)
        y = np.sqrt(2 * (self.y_unwarped - self.epsilon))
        super().__init__(x_init, y, kernel, theta_init, theta_anisotropic, optimize_init)

    def add_data(self, x, y, *args, **kwargs):
        """
        Add data to gp
        :param x: np.ndarray, evaluation point
        :param y: np.ndarray, evaluation value
        """
        self.x = np.vstack((self.x, x))
        self.y_unwarped = np.hstack((self.y, y))
        self.epsilon = 0.8 * np.min(self.y_unwarped)
        self.y = np.sqrt(2 * (self.y_unwarped - self.epsilon))
        self.optimize_theta(*args, **kwargs)

    def predict(self, x):
        """
        Calculate mean and variance of predicted value at queried x point
        :param x: query point
        :return: mean, variance
        """
        gp = self.build_gp().condition(self.y, X_test=x, diag=jnp.exp(self.theta["log_jitter"])).gp
        mean = self.epsilon + 0.5 * gp.loc ** 2
        variance = gp.loc * gp.variance * gp.loc
        return mean, variance

    def plot(self):
        """
        Plot 1d Gaussian process
        :return: None
        """
        lbs = 0
        ubs = 1
        test_x = np.linspace(lbs, ubs, 100).reshape(-1, 1)
        test_y, variance = self.predict(test_x)
        test_sd = jnp.sqrt(variance)


        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(self.x, self.y_unwarped, 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.flatten(), test_y.flatten(), 'b')
        # Shade between the lower and upper confidence bounds, applying softplus to lower bound to smoothly enforce
        # non-negativity
        ax.fill_between(test_x.flatten(), 0.3 * test_y.flatten() *
                        jnp.log(1 + jnp.exp(1 / (0.3 * test_y.flatten()) * (test_y.flatten() - 2 * test_sd))),
                        test_y + 2 * test_sd, alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()
