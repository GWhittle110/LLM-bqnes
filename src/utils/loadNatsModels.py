"""
Function to load models from NATS Bench
"""

import os
from dotenv import dotenv_values
from xautodl.models import get_cell_based_tiny_net

os.environ["HOME"] = dotenv_values()["HOME"]


def load_nats_models(api, indexes: list, dataset: str) -> list:
    """
    Load models from NATS Bench
    :param api: NATS Bench API
    :param indexes: Indexes of model to load from search space
    :param dataset: NATS Bench dataset
    :return: list of models
    """
    def load_single_model(index):
        config = api.get_net_config(index, dataset)
        network = get_cell_based_tiny_net(config)
        params = api.get_net_param(index, dataset, None)
        network.load_state_dict(next(iter(params.values())))
        return network

    return [load_single_model(index) for index in indexes]

