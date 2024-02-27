from yaml import dump
import numpy as np


config_name = "mnistBasicConfig"

config = {"candidate_directory": "mnist_basic",     # Name of directory inside experiments folder containing candidate models
          "dataset": "mnist",                       # Name of file defining dataset objects
          "max_tokens": 100000,                     # Maximum number of tokens to sample from Claude
          "examples": None,                         # Example prompts and replies for Claude
          "task": "Image classification",           # Task model will be used for
          "from_predictions": True,                 # Whether to use pre-calculated predictions
          "train_batch_size": 100,                  # Training batch size
          "test_batch_size": 100,                   # Testing batch size
          "ll_reduction": 1000,                     # Factor to divide log likelihoods by
          "search_space_only": False,               # Whether to proceed once the search space is defined
          "integrand_models": ["IntegrandModel",
                               "IntegrandModel",
                               "IntegrandModel",
                               "IntegrandModel",
                               "SqIntegrandModel",
                               "SqIntegrandModel",
                               "SqIntegrandModel",
                               "SqIntegrandModel",
                               "DiagSqIntegrandModel",
                               "DiagSqIntegrandModel",
                               "DiagSqIntegrandModel",
                               "DiagSqIntegrandModel",
                               "LinSqIntegrandModel",
                               "LinSqIntegrandModel",
                               "LinSqIntegrandModel",
                               "LinSqIntegrandModel"],     # Integrand models to use
          "kernels": ["ExpSquared",
                      "Exp",
                      "Matern32",
                      "Matern52",
                      "ExpSquared",
                      "Exp",
                      "Matern32",
                      "Matern52",
                      "ExpSquared",
                      "Exp",
                      "Matern32",
                      "Matern52",
                      "ExpSquared",
                      "Exp",
                      "Matern32",
                      "Matern52"],                  # Kernels to use
          "log_scales": -1.,                        # Initial kernel log length scales
          "log_amps": 0.,                           # Initial kernel log amplitudes
          "log_jitters": 0.,                        # Initial kernel log jitters
          "theta_anisotropic": True,                # Whether to use an anisotropic length scale for theta
          "optimize_init": False,                   # Whether to optimise model hyperparameters upon initialisation
          "n_acquire": 5,                           # Number of models to acquire for the ensemble (not including init model)
          "min_det": [0, 0.001],                    # Minimum correlation matrix determinant to allow during quadrature
          "test_uniform": True,                     # Whether to also test ensemble with uniform weights and same models
          "nbins": 10                               # Number of bins to use when computing expected calibration error
          }

stream = open(f"experiments\\configs\\{config_name}.yaml", "w+")
dump(config, stream)
