"""
Main file for running experiments
"""

import torchvision
import warnings
from src.LLMSearchSpace import *
from src.bayes_quad import *
from src.utils.logLikelihood import log_likelihood
from src.ensemble import *
from src.utils.loadModels import load_models
from src.utils.accuracy import accuracy

warnings.filterwarnings("ignore", category=UserWarning)

train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

models = load_models("experiments\\mnist")

search_space_constructor = searchSpaceConstructor.AnthropicSearchSpaceConstructor(use_default_examples=False)

search_space = search_space_constructor.construct_search_space(models, "Image classification", train_dataset)

for coordinate in search_space.coordinates:
    search_space.query_log_likelihood(coordinate=coordinate)

integrand = quadrature.IntegrandModel(search_space.coordinates, np.exp(search_space.log_likelihoods / 3000))
evidence, variance = integrand.quad(min_det=0)

print(f"Standard GP Model evidence: {evidence} \u00B1 {2 * np.sqrt(variance)}")

sq_integrand = quadrature.SqIntegrandModel(search_space.coordinates, np.exp(search_space.log_likelihoods / 3000))
sq_evidence, sq_variance = sq_integrand.quad(min_det=0)

print(f"WSABI Model evidence: {sq_evidence} \u00B1 {2 * np.sqrt(sq_variance)}")

uniform_ensemble = Ensemble(search_space.models, np.ones_like(integrand.quad_weights) / len(integrand.quad_weights),
                            np.ones_like(integrand.surrogate.y), 1)
ensemble = Ensemble(search_space.models, integrand.quad_weights, integrand.surrogate.y, evidence)
sq_ensemble = SqEnsemble(search_space.models, sq_integrand.quad_weights, sq_integrand.surrogate.y_unwarped, sq_evidence)
n = len(test_dataset)

print("Test set mean log likelihoods:")
for i, model in enumerate(search_space.models):
    print(f"Model {i+1}: {log_likelihood(model, test_dataset) / n}")
print(f"Uniform weighted ensemble: {log_likelihood(uniform_ensemble, test_dataset) / n}")
print(f"Bayesian quadrature ensemble: {log_likelihood(ensemble, test_dataset) / n}")
print(f"Warped Bayesian quadrature ensemble: {log_likelihood(sq_ensemble, test_dataset) / n}")

print("Test set accuracies:")
for i, model in enumerate(search_space.models):
    print(f"Model {i+1}: {accuracy(model, test_dataset) / n}")
print(f"Uniform weighted ensemble: {accuracy(uniform_ensemble, test_dataset) / n}")
print(f"Bayesian quadrature ensemble: {accuracy(ensemble, test_dataset) / n}")
print(f"Warped Bayesian quadrature ensemble: {accuracy(sq_ensemble, test_dataset) / n}")

