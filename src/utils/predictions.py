"""
Calculate predictions of all models in directory on training and testing datasets and save to pickle
"""
import os
import git
from src.utils.loadModels import load_models
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np


def predictions(directory: str, dataset: str, save: bool = True, batch_size: int = 100) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate and save predictions of all models in directory
    :param directory: Name of directory
    :param dataset: Name of .py file defining dataset objects
    :param save: Whether to save predictions
    :param batch_size: Batch size for calculating predictions
    :return: Dataframe containing training and testing predictions
    """
    device = torch.device("cuda:0")
    models = load_models("experiments\\models\\" + directory)
    dataset_module = __import__("experiments.datasets." + dataset,
                                fromlist=["train_dataset", "test_dataset"])
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def safe_run(model, inputs, device):
        try:
            return model.to(device)(inputs.to(device))
        except AttributeError:
            return model(inputs).to(device)

    print(models)

    with torch.no_grad():
        train_predictions = [torch.cat([safe_run(model, inputs, device)
                                        for inputs, targets in train_dataloader]).cpu().numpy() for model in models]
        test_predictions = [torch.cat([safe_run(model, inputs, device)
                                       for inputs, targets in test_dataloader]).cpu().numpy() for model in models]

    train_df = pd.DataFrame(np.array(train_predictions).transpose((1, 0, 2)).tolist(),
                            columns=[type(model).__name__ for model in models])
    train_df["Target"] = torch.tensor(train_dataset.targets).numpy()
    test_df = pd.DataFrame(np.array(test_predictions).transpose((1, 0, 2)).tolist(),
                           columns=[type(model).__name__ for model in models])
    test_df["Target"] = torch.tensor(test_dataset.targets).numpy()

    if save:
        abs_path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                                "experiments\\models\\" + directory + "\\data\\")
        train_df.to_pickle(os.path.join(abs_path, "train_predictions.pkl"))
        test_df.to_pickle(os.path.join(abs_path, "test_predictions.pkl"))

    return train_df, test_df

