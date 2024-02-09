"""
Main file for running experiments
"""

import numpy as np
import torchvision
import torch
import inspect
from src.LLMSearchSpace import *
from src.bayes_quad import *
import mnistEnsembleExample.cnn1

torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

models = [eval(f"mnistEnsembleExample.{model}.{model.upper()}()")
          for model, _ in inspect.getmembers(mnistEnsembleExample)
          if "train" not in model and "__" not in model]

