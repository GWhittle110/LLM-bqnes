"""
Module containing class for managing a Gaussian process and carrying out Bayesian quadrature
"""
import scipy.optimize
import torch
import gpytorch
from scipy.stats import qmc
import matplotlib.pyplot as plt


class ExactQuadGP(gpytorch.models.ExactGP):

    def __init__(self, x_init: torch.Tensor = None,
                 y_init: torch.Tensor = None,
                 ndim: int = None,
                 mean_module=gpytorch.means.ConstantMean(),
                 covar_module=gpytorch.kernels.RBFKernel(),
                 likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        """
        Gaussian process to carry out Bayesian quadrature using exact inference
        :param x_init: Input points used to initialise the GP
        :param y_init: Output point used to initialise the GP
        :param ndim: Dimensionality of input data. Redundant if x_init provided
        :param mean_module: Prior mean function, by default set to constant zero
        :param covar_module: Covariance function, by default set to negative exponential square radial basis function
        :param likelihood: Likelihood function over GP, by default a Gaussian likelihood
        Example:
        >>> x_init = ExactQuadGP.get_init_points(5, 1)
        >>> y_init = x_init**2
        >>> GP = ExactQuadGP(x_init, y_init)
        """
        super(ExactQuadGP, self).__init__(x_init, y_init, likelihood)

        if x_init is None:
            self.x = torch.empty(ndim)
            self.y = torch.empty(1)
            self.ndim = ndim
        else:
            self.x = x_init
            self.y = y_init.reshape(-1)
            self.ndim = x_init.shape[1]

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood

        self.set_train_data(self.x, self.y)

        self.covar_module.lengthscale = 0.5
        self.likelihood.noise = 0.0001
        self.quad_calculated = False

        self.eval()
        self.likelihood.eval()

    def forward(self, x: torch.Tensor):
        """
        Evaluate GP at input x and return corresponding probability distribution
        :param x: Input tensor
        :return: Conditional distribution induced by GP at x
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def acquire_next(self, init_x: torch.Tensor, method: str = "Variance"):
        """
        Evaluate acquisition function to determine next point to sample
        :param method: str, objective function to be maximised. Defaults to variance
        :param init_x: torch.Tensor, point to initialise minimisation routine
        :return: torch.Tensor, next sample point
        Example:
        >>> lbs = -2
        >>> ubs = 2
        >>> x = ExactQuadGP.get_init_points(3, 1, lbs, ubs)
        >>> y = torch.exp(-x**2).reshape(-1)
        >>> quad_gp = ExactQuadGP(x, y)
        >>> quad_gp.plot_gp(lbs, ubs)
        >>> x_next = quad_gp.acquire_next(torch.tensor(1))
        >>> y_next = torch.exp(-x_next**2)
        >>> print(f'Acquired sample: {x_next}, corresponding function evaluation: {y_next}')
        """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if method == "Variance":
                objective = lambda x: -1 * self.likelihood(self(torch.tensor(float(x)).reshape(-1))).variance.numpy()
            else:
                raise NotImplementedError
            res = scipy.optimize.minimize(objective, init_x.numpy())
            print(res)
            return torch.tensor(res.x)

    def quad(self, lbs: list, ubs: list):
        """
        Calculate integral and quadrature weights for used quadrature points
        :param lbs: lower bounds for integration
        :param ubs: upper bounds for integration
        :return: integral result
        Example:
        >>> lbs = -10
        >>> ubs = 10
        >>> x = ExactQuadGP.get_init_points(10, 1, lbs, ubs)
        >>> y = torch.exp(-x**2)
        >>> quad_gp = ExactQuadGP(x, y)
        >>> print(quad_gp.quad(lbs, ubs))
        >>> print(f'Quadrature points: {quad_gp.x.reshape(-1)}, Weights: {quad_gp.quad_weights}')
        """
        if self.quad_calculated:
            return self.quad_weights @ self.y
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                integrand = lambda x, X: self.covar_module(torch.tensor(x).reshape(-1), X).numpy()
                if self.ndim == 1:
                    KxX = torch.tensor([scipy.integrate.quad(lambda x: integrand(x, X), lbs, ubs)[0] for X in self.x])
                else:
                    KxX = torch.tensor([scipy.integrate.nquad(lambda *x: integrand(x, X), *zip(lbs, ubs))[0] for X in self.x])
                KXX = self.covar_module(self.x, self.x)
                KXX = self.add_jitter(KXX, min_det=0.1)
                self.quad_weights = gpytorch.solve(KXX, KxX)
                self.quad_calculated = True
                return self.quad_weights @ self.y

    def plot_gp(self, lbs, ubs):
        """
        Plot 1d Gaussian process
        :return: None
        """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lbs, ubs, 100)
            observed_pred = self.likelihood(self(test_x))

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(self.x.numpy(), self.y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()

    @staticmethod
    def add_jitter(mat: torch.Tensor, min_det: float = None, jitter_val: float = 0.01):
        """
        Function to add jitter to covariance matrices for better numerical conditioning
        :param mat: The matrix to condition
        :param min_det: Determinant at which to stop adding jitter
        :param jitter_val: Amount of jitter to add each iteration
        :return: The conditioned matrix
        Example:
        >>> matrix = torch.tensor([[1,0.999],[0.999,1]])
        >>> print(torch.det(matrix))
        >>> min_det = 0.1
        >>> jitter_matrix = ExactQuadGP.add_jitter(matrix, min_det)
        >>> print(jitter_matrix)
        >>> print(torch.det(jitter_matrix))
        """
        if min_det is None:
            return gpytorch.add_jitter(mat, jitter_val)
        try:
            while torch.det(mat) < min_det:
                mat = gpytorch.add_jitter(mat, jitter_val)
        except NotImplementedError:
            return gpytorch.add_jitter(mat, jitter_val)
        return mat

    @staticmethod
    def get_init_points(n_points: int, ndim: int, lbs: list = None, ubs: list = None):
        """
        Helper function to generate initialisation points using Latin hypercube
        :param n_points: Number of points to sample
        :param ndim: Number of dimensions to sample in
        :param lbs: Lower bounds for rescaling points
        :param ubs: Upper bounds for rescaling points
        :return: Tensor of initialisation points
        Example:
        >>> n_points = 5
        >>> ndim = 3
        >>> lbs = [-1, -1, -1]
        >>> ubs = [1, 1, 1]
        >>> samples = ExactQuadGP.get_init_points(n_points, ndim, lbs, ubs)
        >>> print(samples)
        """
        if lbs is None:
            lbs = [0] * ndim
        if ubs is None:
            ubs = [1] * ndim
        sampler = qmc.LatinHypercube(d=ndim)
        sample = sampler.random(n_points)
        sample_scaled = qmc.scale(sample, lbs, ubs)
        return torch.tensor(sample_scaled, dtype = torch.float)


