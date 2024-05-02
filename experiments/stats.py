from json import load
import numpy as np
from yaml import dump


def stats(log_dir: str, log_number: int):
    """
    Function to evaluate stats for experiment runs
    :param log_dir: Name of directory containing specific logs
    :param log_number: Number of log required
    :return: Run stats
    """

    def tuple_key_parser(key_tuple):
        # Change number at end to correct number for specific use case
        return 'json://{"py/tuple": [{"py/tuple": ["' + '", "'.join([key for key in key_tuple]) + '"]}, 45]}'

    with open(f"./logs/{log_dir}/{log_number}/info.json") as f:
        info = load(f)

    with open(f"./logs/{log_dir}/{log_number}/config.json") as f:
        config = load(f)

    def process_subdict(subdict, no_ensemble=False):
        if not no_ensemble:
            ll = np.array(subdict["ensemble_log_likelihood"])
            mask = ~(np.isinf(ll) | np.isnan(ll) | (ll < -12100))
        else:
            mask = np.ones(10).astype(bool)

        if no_ensemble:
            fields = [
            "uniform_ensemble_log_likelihood", "uniform_ensemble_accuracy", "uniform_ensemble_expected_calibration_error",
            "bayes_ensemble_log_likelihood", "bayes_ensemble_accuracy", "bayes_ensemble_expected_calibration_error"
            ]
        else:
            fields = ["ensemble_log_likelihood", "ensemble_accuracy", "ensemble_expected_calibration_error",
                      "uniform_ensemble_log_likelihood", "uniform_ensemble_accuracy", "uniform_ensemble_expected_calibration_error",
                      "bayes_ensemble_log_likelihood", "bayes_ensemble_accuracy", "bayes_ensemble_expected_calibration_error"]
        return {field: {"mean": np.mean(np.array(subdict[field])[mask]).tolist(),
                        "std": np.std(np.array(subdict[field])[mask]).tolist()}
                for field in fields}



    stats_dict = {tuple_key: process_subdict(info[tuple_key_parser(tuple_key)])
                  for tuple_key in zip(config["integrand_models"], config["kernels"])}

    if config["test_random"]:
        stats_dict["random"] = process_subdict(info['json://{\"py/tuple\": [\"random\", 45]}'], no_ensemble=True)

    with open(f"./logs/{log_dir}/{log_number}/stats.yaml", "w+") as f:
        dump(stats_dict, f)

    return stats_dict


info = stats("nats_ImageNet16-120", 14)



