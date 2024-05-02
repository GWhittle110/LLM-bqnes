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
import numpy as np
import torch


path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                    "experiments/configs/natsCIFAR10Config.yaml")

datasets = ["ImageNet16-120"]

data_dict = dict()
for dataset in datasets:

    with open(path, "rb") as f:
        nats_indexes = safe_load(f)["nats_indexes"]

    api = create(dotenv_values()["HOME"] + "\\.torch\\NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)

    predictions_path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                                    f'experiments/models/nats_{dataset}/data/')
    test_predictions = pd.read_pickle(os.path.join(predictions_path, "test_predictions.pkl"))

    infos = [api.get_more_info(index, dataset, hp=200) for index in nats_indexes]

    test_losses = [info["test-loss"] for info in infos]

    dataset_data_dict = dict()
    dataset_data_dict["index"] = nats_indexes[np.argmin(test_losses).item()]
    dataset_data_dict["ll"] = -1 * np.min(test_losses).item() * len(test_predictions)
    dataset_data_dict["accuracy"] = infos[np.argmin(test_losses).item()]["test-accuracy"]
    dataset_data_dict["ece"] = expected_calibration_error_from_predictions(torch.tensor(test_predictions[dataset_data_dict["index"]]),
                                                             test_predictions["Target"].values, 10).item()

    data_dict[dataset] = dataset_data_dict

with open("nats_best_single.yaml", "w+") as f:
    dump(data_dict, f)

