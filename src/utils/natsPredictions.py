"""
Calculate predictions of all models in NATS Bench search space on testing dataset and save to pickle
"""
import os
import git
from src.utils.loadNatsModels import load_nats_models
from src.utils.loadNatsStateDict import load_nats_state_dict
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import Softmax
import numpy as np
from yaml import safe_load
from nats_bench import create
from dotenv import dotenv_values
from xautodl.datasets import get_datasets


def nats_predictions(api, nats_indexes: list, dataset: str, save_directory: str = None, batch_size: int = 100
                     ) -> pd.DataFrame:
    """
    Calculate and save predictions of all models in directory
    :param api: NATS
    :param nats_indexes: List containing indexes of nats models
    :param dataset: Name of dataset
    :param save_directory: Where to save predictions to
    :param batch_size: Batch size for calculating predictions
    :return: Dataframe containing training and testing predictions
    """
    path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                        f"experiments/datasets/data/{dataset}")
    _, test_dataset, _, _ = get_datasets(dataset, path, -1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if dataset == "ImageNet16-120":
        test_dataset.targets = [target - 1 for target in test_dataset.targets]

    device = torch.device("cuda:0")
    models = load_nats_models(api, nats_indexes, dataset)
    models_combined = [(load_nats_state_dict(api, model, index, dataset), index)
                       for model, index in zip(models, nats_indexes)]

    def safe_run(model, inputs, device):
        try:
            return model.to(device)(inputs.to(device))
        except AttributeError:
            return model.model(inputs).to(device)

    with torch.no_grad():
        sm = Softmax()
        test_predictions = [torch.cat([sm(safe_run(model, inputs, device)[1])
                                       for inputs, targets in test_dataloader]).cpu().numpy()
                            for model, index in models_combined]

    test_df = pd.DataFrame(np.array(test_predictions).transpose((1, 0, 2)).tolist(),
                           columns=[index for index in nats_indexes])
    test_df["Target"] = torch.tensor(test_dataset.targets).numpy()

    if save_directory is not None:
        abs_path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                                "experiments\\models\\" + save_directory + "\\data\\")
        test_df.to_pickle(os.path.join(abs_path, "test_predictions.pkl"))

    return test_df


if __name__ == "__main__":
    os.environ["TORCH_HOME"] = dotenv_values()["HOME"] + "\\.torch"

    path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                        "experiments/configs/natsCIFAR10Config.yaml")

    with open(path, "rb") as f:
        nats_indexes = safe_load(f)["nats_indexes"]

    api = create(dotenv_values()["HOME"] + "\\.torch\\NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
    dataset = "ImageNet16-120"
    save_directory = "nats_" + dataset
    dataset = nats_predictions(api, nats_indexes, dataset, save_directory)



