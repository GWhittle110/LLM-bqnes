"""
Function to load state dict for NATS Bench model
"""


def load_nats_state_dict(api, network, index: list, dataset: str):
    """
    Load models from NATS Bench
    :param api: NATS Bench API
    :param network: network object
    :param index: Index of model to load from search space
    :param dataset: NATS Bench dataset
    :return: list of models
    """
    params = api.get_net_param(index, dataset, None)
    network.load_state_dict(next(iter(params.values())))
    return network.eval()
