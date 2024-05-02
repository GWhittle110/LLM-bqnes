"""
Calculate best single model in NATS search space
"""

from nats_bench import create
import os
import git
from dotenv import dotenv_values
from yaml import safe_load, dump
import pandas as pd
from src.utils.expectedCalibrationError import expected_calibration_error_from_predictions
from src.utils.accuracy import accuracy_from_predictions
from src.utils.logLikelihood import log_likelihood_from_predictions
from src.utils.loadModels import load_models
import numpy as np
import torch


model_dirs = ["mnist_basic", "cifar_basic"]

data_dict = dict()
for model_dir in model_dirs:

    predictions_path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                                    f'experiments/models/{model_dir}/data/')
    test_predictions = pd.read_pickle(os.path.join(predictions_path, "test_predictions.pkl"))
    candidate_models = load_models(f"experiments\\models\\{model_dir}")
    model_names = [type(model).__name__ for model in candidate_models]
    test_lls = [log_likelihood_from_predictions(test_predictions[name], test_predictions["Target"].values) for name in model_names]

    dataset_data_dict = dict()
    dataset_data_dict["model"] = model_names[np.argmax(test_lls).item()]
    dataset_data_dict["ll"] = np.max(test_lls).item()
    dataset_data_dict["accuracy"] = accuracy_from_predictions(torch.tensor(test_predictions[dataset_data_dict["model"]]),
                                                             test_predictions["Target"].values)
    dataset_data_dict["ece"] = expected_calibration_error_from_predictions(torch.tensor(test_predictions[dataset_data_dict["model"]]),
                                                             test_predictions["Target"].values, 10).item()

    data_dict[model_dir] = dataset_data_dict

with open("best_single.yaml", "w+") as f:
    dump(data_dict, f)

