"""
Example implementation of Bayesian quadrature using GPyTorch library for the GP integrand surrogate
"""
import scipy.integrate
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np


# Define integrand function and randomly sample some initial points for parameters
def integrand(x, no_torch=False):
    if no_torch:
        x = torch.tensor(x)
    out = torch.sinc(x)
    if no_torch:
        out = out.numpy()
    return out


n_init = 5
domain = [-5, 5]
x_sample = torch.linspace(domain[0], domain[1], n_init)
y_sample = integrand(x_sample)


# Define GP prior for integrand
class GP(gpytorch.models.ExactGP):

    def __init__(self, x_sample, y_sample, likelihood):
        super(GP, self).__init__(x_sample, y_sample, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Function for plotting GP "Sausage plot"
def plot_gp(model, likelihood, x_sample):
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(domain[0], domain[1], 100)
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(x_sample.numpy(), integrand(x_sample).numpy(), 'k*')
        ax.plot(test_x.numpy(), integrand(test_x).numpy())
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'True function', 'Mean', 'Confidence'])
        plt.show()


# Acquisition function for finding next point to query, using max variance targeter
def acquire_next_point(model, likelihood):
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(domain[0], domain[1], 100)
        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()
        uncertainty = upper - lower

        # Return coordinate corresponding to maximum uncertainty
        return test_x[torch.argmax(uncertainty)]


# Define likelihood function, initialise model and manually define hyperparameters
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GP(x_sample, y_sample, likelihood)
model.covar_module.lengthscale = 0.5
likelihood.noise = 0.0001

# Display model with this data
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
plot_gp(model, likelihood, x_sample)

# Sequentially acquire new evaluation points
n_samples = 10
for i in range(n_samples):
    x_sample = torch.cat((x_sample, acquire_next_point(model, likelihood).reshape(-1)))
    model.set_train_data(x_sample, integrand(x_sample), strict=False)
    if not bool((i+1) % 5):
        plot_gp(model, likelihood, x_sample)

# Calculate approximate integral in form of quadrature rule
with torch.no_grad():
    dx = 0.5
    x_test = torch.range(domain[0], domain[1], dx)

    KXX = torch.tensor([[model.covar_module.forward(x1, x2) for x1 in x_sample] for x2 in x_sample])
    inv_KXX = torch.inverse(KXX)

    y_sample = integrand(x_sample)
    KxX = torch.zeros(x_sample.shape[0])
    for x in x_test:
        KxX += dx * torch.tensor([model.covar_module.forward(x, xdash) for xdash in x_sample])

    weights = KxX @ inv_KXX
    integral = weights @ y_sample
    true = scipy.integrate.quad(lambda x: integrand(x, True), domain[0], domain[1])[0]

    print(f'Bayesian Quadrature Complete - integral value: {integral} \n percentage error: {(integral.numpy()-true) / true * 100}% \n quadrature points: {x_sample} \n quadrature weights: {weights}')
