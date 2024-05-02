"""
Main file for running experiments
"""

import warnings
import numpy as np
from src.LLMSearchSpace import *
from src.bayes_quad import *
from src.utils.logLikelihood import log_likelihood, log_likelihood_from_predictions
from src.ensemble import *
from src.utils.loadModels import load_models
from src.utils.loadNatsModels import load_nats_models
from src.utils.expectedCalibrationError import expected_calibration_error, expected_calibration_error_from_predictions
from src.utils.accuracy import accuracy, accuracy_from_predictions
from src.utils.dictCat import dict_cat
from tinygp.kernels.stationary import Exp, ExpSquared, Matern32, Matern52
import logging
from src.utils.getModels import get_models
from sacred import Experiment
from sacred.observers import FileStorageObserver
from src.utils.zipper import zipper
from importlib import __import__
import os
import git
import pandas as pd
import yaml
from nats_bench import create
from dotenv import dotenv_values
from xautodl.datasets import get_datasets

os.environ["TORCH_HOME"] = dotenv_values()["HOME"] + "\\.torch"
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=20)
logger = logging.getLogger("main")

# Create sacred experiment and set up observers and config
config_name = "natsImageNet16-120Config"
ex = Experiment()
ex.add_config(f"..\\experiments\\configs\\{config_name}.yaml")
with open(f"..\\experiments\\configs\\{config_name}.yaml", 'r') as file:
    config = yaml.safe_load(file)
ex.observers.append(FileStorageObserver('..\\experiments\\logs\\' + config["candidate_directory"]))

@ex.automain
def run_experiment(_config=None, _run=None):
    """
    Run an experiment
    :param _config: Experiment config, autofilled by sacred
    :param _run: Sacred run object
    """

    nats = "nats_indexes" in _config
    if nats:
        # Initialise nats models
        api = create(dotenv_values()["HOME"] + "\\.torch\\NATS-tss-v1_0-3ffb9-full", _config["nats_ss"],
                     fast_mode=True, verbose=True)
        nats_info = {
            "api": api,
            "dataset": _config["dataset"],
            "nats_indexes": _config["nats_indexes"]
        }
        candidate_models = load_nats_models(api, _config["nats_indexes"], _config["dataset"])
    else:
        # Get the candidate models from the experiment model directory
        nats_info = None
        candidate_models = load_models("experiments\\models\\"+_config["candidate_directory"])
        logger.info(f"Candidate Models: {[type(model).__name__ for model in candidate_models]}")

    # Load dataset
    if not nats:
        dataset_module = __import__("experiments.datasets."+_config["dataset"], fromlist=["train_dataset", "test_dataset"])
        train_dataset = getattr(dataset_module, "train_dataset")
        test_dataset = getattr(dataset_module, "test_dataset")
    else:
        path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                            f'experiments/datasets/data/{_config["dataset"]}')
        train_dataset, test_dataset, _, _ = get_datasets(_config["dataset"], path, -1)

        if _config["dataset"] == "ImageNet16-120":
            train_dataset.targets = [target - 1 for target in train_dataset.targets]
            test_dataset.targets = [target - 1 for target in test_dataset.targets]

    logger.info(f"Train set length: {len(train_dataset)}")
    logger.info(f"Test set length: {len(test_dataset)}")

    # Load predictions
    if _config["from_predictions"]:
        path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                            "experiments\\models\\" + _config["candidate_directory"] + "\\data\\")
        if not nats:
            train_predictions = pd.read_pickle(os.path.join(path, "train_predictions.pkl"))
        else:
            train_predictions = None
        test_predictions = pd.read_pickle(os.path.join(path, "test_predictions.pkl"))
    else:
        train_predictions = None
        test_predictions = None

    # Repeat runs from here unless NATS bench
    repeats_from_start = 1 if nats else _config["repeats"]
    for i in range(repeats_from_start):
        logger.info(f"Beginning total repeat: {i}")

        # Construct the search space object
        search_space_constructor = searchSpaceConstructor.AnthropicSearchSpaceConstructor(max_tokens=_config["max_tokens"],
                                                                                          examples=_config["examples"])
        search_space, discrete_dims = search_space_constructor.construct_search_space(models=candidate_models,
                                                                                      task=_config["task"],
                                                                                      dataset=train_dataset,
                                                                                      predictions=train_predictions,
                                                                                      reduction_factor=_config["ll_reduction"],
                                                                                      nats_info=nats_info,
                                                                                      save_prompt=_config["save_prompt"],
                                                                                      _run=_run)

        # Early stop if specified
        if _config["search_space_only"]:
            return

        # initialise the integrands requested as a dict keyed on the integrand model - kernel pairs
        integrand_models_dict = {"IntegrandModel": quadrature.IntegrandModel,
                                 "SqIntegrandModel": quadrature.SqIntegrandModel,
                                 "DiagSqIntegrandModel": quadrature.DiagSqIntegrandModel,
                                 "LinSqIntegrandModel": quadrature.LinSqIntegrandModel}

        kernels_dict = {"Exp": Exp,
                        "ExpSquared": ExpSquared,
                        "Matern32": Matern32,
                        "Matern52": Matern52}

        acquisitions_dict = {"IntegrandModel": acquisitions.DiscreteWarpedUncertaintySampling,
                             "SqIntegrandModel": acquisitions.DiscreteUncertaintySampling,
                             "DiagSqIntegrandModel": acquisitions.DiscreteUncertaintySampling,
                             "LinSqIntegrandModel": acquisitions.DiscreteUncertaintySampling}

        theta_inits = [{"log_scale": np.float64(log_scale),
                        "log_amp": np.float64(log_amp),
                        "log_jitter": np.float64(log_jitter)}
                       for log_scale, log_amp, log_jitter
                       in zipper(_config["log_scales"], _config["log_amps"], _config["log_jitters"])]

        repeats_from_middle = 1 if not nats else _config["repeats"]
        for j in range(repeats_from_middle):
            logger.info(f"Beginning calculation repeat: {j}")

            # Choose a random model(s) to initialise on
            init_candidates = np.arange(len(search_space.log_likelihoods))
            init_indexes = np.random.choice(init_candidates, size=min(_config["n_init"],
                                                                      len(search_space.log_likelihoods)), replace=False)
            for index in init_indexes:
                search_space.query_log_likelihood(index)

            integrands = {(integrand_model, kernel): integrand_models_dict[integrand_model](search_space.coordinates[init_indexes].reshape(_config["n_init"], -1),
                                                                                            np.exp(search_space.log_likelihoods[init_indexes]),
                                                                                            kernel=kernels_dict[kernel],
                                                                                            theta_init=theta_init,
                                                                                            theta_anisotropic=_config["theta_anisotropic"],
                                                                                            optimize_init=optimize_init)
                          for integrand_model, kernel, theta_init, optimize_init
                          in zipper(_config["integrand_models"], _config["kernels"], theta_inits, _config["optimize_init"])}

            # Henceforth repeat experiment for all integrands
            for (name, integrand), n_acquire, min_det, test_uniform, test_bayes in zipper(integrands.items(), _config["n_acquire"], _config["min_det"], _config["test_uniform"], _config["test_bayes"]):
                data_dict = dict()

                logger.info(f"Evaluating {name[0]} using {name[1]} kernel \n")

                # Acquire models coordinates
                acquisition = acquisitions_dict[name[0]](integrand.surrogate, search_space, epsilon=0.01)
                acquisition.acquire(n_acquire, _config["train_batch_size"])
                models_used = get_models(integrand, search_space)
                if nats:
                    data_dict["models"] = {model[1]: i for i, model in enumerate(models_used)}
                else:
                    data_dict["models"] = {type(model).__name__: i for i, model in enumerate(models_used)}

                # Run quadrature routine
                evidence, variance = integrand.quad(min_det=min_det, discrete_dims=discrete_dims)
                theta = integrand.surrogate.theta
                data_dict["theta"] = {"log_scale": theta["log_scale"].tolist(),
                                      "log_amp": theta["log_scale"].tolist(),
                                      "log_jitter": theta["log_scale"].tolist()}
                data_dict["evidence"] = float(evidence)
                data_dict["variance"] = float(variance)

                logger.info(f"Model evidence: {evidence} \u00B1 {2 * np.sqrt(variance)} \n")

                # Construct ensemble
                ensemble_dict = {"IntegrandModel": Ensemble,
                                 "SqIntegrandModel": SqEnsemble,
                                 "DiagSqIntegrandModel": DiagSqEnsemble,
                                 "LinSqIntegrandModel": LinSqEnsemble}

                ensemble = ensemble_dict[name[0]](models_used, integrand, nats)

                # Evaluate ensemble log likelihood, accuracy and expected calibration error
                if test_predictions is not None:
                    predictions = ensemble.forward_from_predictions(test_predictions).numpy()
                    ensemble_log_likelihood = log_likelihood_from_predictions(predictions, test_predictions["Target"])
                    data_dict["ensemble_log_likelihood"] = ensemble_log_likelihood
                    logger.info(f"Ensemble log likelihood: {ensemble_log_likelihood}")
                    ensemble_accuracy = accuracy_from_predictions(predictions, test_predictions["Target"])
                    data_dict["ensemble_accuracy"] = ensemble_accuracy
                    logger.info(f"Ensemble accuracy: {ensemble_accuracy}")
                    ensemble_ece = expected_calibration_error_from_predictions(predictions, test_predictions["Target"],
                                                                               _config["nbins"])
                    data_dict["ensemble_expected_calibration_error"] = float(ensemble_ece)
                    logger.info(f"Ensemble expected calibration error: {ensemble_ece} \n")
                else:
                    ensemble_log_likelihood = log_likelihood(ensemble, test_dataset, _config["test_batch_size"])
                    data_dict["\n ensemble_log_likelihood"] = ensemble_log_likelihood
                    logger.info(f"Ensemble log likelihood: {ensemble_log_likelihood}")
                    ensemble_accuracy = accuracy(ensemble, test_dataset, _config["test_batch_size"])
                    data_dict["ensemble_accuracy"] = ensemble_accuracy
                    logger.info(f"Ensemble accuracy: {ensemble_accuracy}")
                    ensemble_ece = expected_calibration_error(ensemble, test_dataset, _config["nbins"],
                                                              _config["test_batch_size"])
                    data_dict["ensemble_expected_calibration_error"] = float(ensemble_ece)
                    logger.info(f"Ensemble expected calibration error: {ensemble_ece} \n")

                # If required, also evaluate ensemble with uniform weights
                if test_uniform:
                    uniform_ensemble = UniformEnsemble(models_used, nats)
                    if test_predictions is not None:
                        uniform_predictions = uniform_ensemble.forward_from_predictions(test_predictions).numpy()
                        uniform_ensemble_log_likelihood = log_likelihood_from_predictions(uniform_predictions,
                                                                                          test_predictions["Target"])
                        data_dict["uniform_ensemble_log_likelihood"] = uniform_ensemble_log_likelihood
                        logger.info(f"Uniform ensemble log likelihood: {uniform_ensemble_log_likelihood}")
                        uniform_ensemble_accuracy = accuracy_from_predictions(uniform_predictions, test_predictions["Target"])
                        data_dict["uniform_ensemble_accuracy"] = uniform_ensemble_accuracy
                        logger.info(f"Uniform ensemble accuracy: {uniform_ensemble_accuracy}")
                        uniform_ensemble_ece = expected_calibration_error_from_predictions(uniform_predictions,
                                                                                           test_predictions["Target"],
                                                                                           _config["nbins"])
                        data_dict["uniform_ensemble_expected_calibration_error"] = float(uniform_ensemble_ece)
                        logger.info(f"Uniform ensemble expected calibration error: {uniform_ensemble_ece} \n")
                    else:
                        uniform_ensemble_log_likelihood = log_likelihood(uniform_ensemble, test_dataset,
                                                                         _config["test_batch_size"])
                        data_dict["uniform_ensemble_log_likelihood"] = uniform_ensemble_log_likelihood
                        logger.info(f"Uniform ensemble log likelihood: {uniform_ensemble_log_likelihood}")
                        uniform_ensemble_accuracy = accuracy(uniform_ensemble, test_dataset,
                                                             _config["test_batch_size"])
                        data_dict["uniform_ensemble_accuracy"] = uniform_ensemble_accuracy
                        logger.info(f"Uniform ensemble accuracy: {uniform_ensemble_accuracy}")
                        uniform_ensemble_ece = expected_calibration_error(uniform_ensemble, test_dataset, _config["nbins"],
                                                                  _config["test_batch_size"])
                        data_dict["uniform_ensemble_expected_calibration_error"] = float(uniform_ensemble_ece)
                        logger.info(f"Uniform ensemble expected calibration error: {uniform_ensemble_ece} \n")
                else:
                    data_dict["uniform_ensemble_loss"] = None
                    data_dict["uniform_ensemble_accuracy"] = None
                    data_dict["uniform_ensemble_expected_calibration_error"] = None

                # If required, also evaluate ensemble with likelihood weights
                if test_bayes:
                    bayes_ensemble = BayesEnsemble(models_used, integrand, nats)
                    if test_predictions is not None:
                        bayes_predictions = bayes_ensemble.forward_from_predictions(test_predictions).numpy()
                        bayes_ensemble_log_likelihood = log_likelihood_from_predictions(bayes_predictions,
                                                                                          test_predictions["Target"])
                        data_dict["bayes_ensemble_log_likelihood"] = bayes_ensemble_log_likelihood
                        logger.info(f"Bayes ensemble log likelihood: {bayes_ensemble_log_likelihood}")
                        bayes_ensemble_accuracy = accuracy_from_predictions(bayes_predictions,
                                                                              test_predictions["Target"])
                        data_dict["bayes_ensemble_accuracy"] = bayes_ensemble_accuracy
                        logger.info(f"Bayes ensemble accuracy: {bayes_ensemble_accuracy}")
                        bayes_ensemble_ece = expected_calibration_error_from_predictions(bayes_predictions,
                                                                                           test_predictions["Target"],
                                                                                           _config["nbins"])
                        data_dict["bayes_ensemble_expected_calibration_error"] = float(bayes_ensemble_ece)
                        logger.info(f"Bayes ensemble expected calibration error: {bayes_ensemble_ece} \n")
                    else:
                        bayes_ensemble_log_likelihood = log_likelihood(bayes_ensemble, test_dataset,
                                                                         _config["test_batch_size"])
                        data_dict["bayes_ensemble_log_likelihood"] = bayes_ensemble_log_likelihood
                        logger.info(f"Bayes ensemble log likelihood: {bayes_ensemble_log_likelihood}")
                        bayes_ensemble_accuracy = accuracy(bayes_ensemble, test_dataset,
                                                             _config["test_batch_size"])
                        data_dict["bayes_ensemble_accuracy"] = bayes_ensemble_accuracy
                        logger.info(f"Bayes ensemble accuracy: {bayes_ensemble_accuracy}")
                        bayes_ensemble_ece = expected_calibration_error(bayes_ensemble, test_dataset, _config["nbins"],
                                                                          _config["test_batch_size"])
                        data_dict["bayes_ensemble_expected_calibration_error"] = float(bayes_ensemble_ece)
                        logger.info(f"Bayes ensemble expected calibration error: {bayes_ensemble_ece} \n")
                else:
                    data_dict["bayes_ensemble_loss"] = None
                    data_dict["bayes_ensemble_accuracy"] = None
                    data_dict["bayes_ensemble_expected_calibration_error"] = None

                if (name, len(models_used)) in _run.info:
                    _run.info[(name, len(models_used))] = dict_cat(_run.info[(name, len(models_used))], data_dict)
                else:
                    _run.info[(name, len(models_used))] = data_dict

            # If required, also evaluate random ensemble with uniform weights and likelihood weights
            if _config["test_random"]:
                logger.info("Testing random ensemble")
                data_dict = dict()

                # Choose random models
                rand_candidates = np.arange(len(search_space.log_likelihoods))
                rand_indexes = np.random.choice(rand_candidates, size=min(_config["n_init"]+_config["n_acquire"],
                                                                          len(search_space.log_likelihoods)), replace=False)
                for index in rand_indexes:
                    search_space.query_log_likelihood(index)

                integrand = IntegrandModel(search_space.coordinates[rand_indexes].reshape(_config["n_init"] +
                                                                                          _config["n_acquire"], -1),
                                           np.exp(search_space.log_likelihoods[rand_indexes]),
                                           kernel=Matern52,
                                           theta_init=theta_inits[0],
                                           theta_anisotropic=_config["theta_anisotropic"],
                                           optimize_init=True)

                integrand.quad(min_det=0.1, discrete_dims=discrete_dims)

                models_used = get_models(integrand, search_space)

                # Test uniform weights
                uniform_ensemble = UniformEnsemble(models_used, nats)
                if test_predictions is not None:
                    uniform_predictions = uniform_ensemble.forward_from_predictions(test_predictions).numpy()
                    uniform_ensemble_log_likelihood = log_likelihood_from_predictions(uniform_predictions,
                                                                                      test_predictions["Target"])
                    data_dict["uniform_ensemble_log_likelihood"] = uniform_ensemble_log_likelihood
                    logger.info(f"Uniform ensemble log likelihood: {uniform_ensemble_log_likelihood}")
                    uniform_ensemble_accuracy = accuracy_from_predictions(uniform_predictions, test_predictions["Target"])
                    data_dict["uniform_ensemble_accuracy"] = uniform_ensemble_accuracy
                    logger.info(f"Uniform ensemble accuracy: {uniform_ensemble_accuracy}")
                    uniform_ensemble_ece = expected_calibration_error_from_predictions(uniform_predictions,
                                                                                       test_predictions["Target"],
                                                                                       _config["nbins"])
                    data_dict["uniform_ensemble_expected_calibration_error"] = float(uniform_ensemble_ece)
                    logger.info(f"Uniform ensemble expected calibration error: {uniform_ensemble_ece} \n")
                else:
                    uniform_ensemble_log_likelihood = log_likelihood(uniform_ensemble, test_dataset,
                                                                     _config["test_batch_size"])
                    data_dict["uniform_ensemble_log_likelihood"] = uniform_ensemble_log_likelihood
                    logger.info(f"Uniform ensemble log likelihood: {uniform_ensemble_log_likelihood}")
                    uniform_ensemble_accuracy = accuracy(uniform_ensemble, test_dataset,
                                                         _config["test_batch_size"])
                    data_dict["uniform_ensemble_accuracy"] = uniform_ensemble_accuracy
                    logger.info(f"Uniform ensemble accuracy: {uniform_ensemble_accuracy}")
                    uniform_ensemble_ece = expected_calibration_error(uniform_ensemble, test_dataset, _config["nbins"],
                                                                      _config["test_batch_size"])
                    data_dict["uniform_ensemble_expected_calibration_error"] = float(uniform_ensemble_ece)
                    logger.info(f"Uniform ensemble expected calibration error: {uniform_ensemble_ece} \n")

                # Test likelihood weights
                bayes_ensemble = BayesEnsemble(models_used, integrand, nats)
                if test_predictions is not None:
                    bayes_predictions = bayes_ensemble.forward_from_predictions(test_predictions).numpy()
                    bayes_ensemble_log_likelihood = log_likelihood_from_predictions(bayes_predictions,
                                                                                    test_predictions["Target"])
                    data_dict["bayes_ensemble_log_likelihood"] = bayes_ensemble_log_likelihood
                    logger.info(f"Bayes ensemble log likelihood: {bayes_ensemble_log_likelihood}")
                    bayes_ensemble_accuracy = accuracy_from_predictions(bayes_predictions,
                                                                        test_predictions["Target"])
                    data_dict["bayes_ensemble_accuracy"] = bayes_ensemble_accuracy
                    logger.info(f"Bayes ensemble accuracy: {bayes_ensemble_accuracy}")
                    bayes_ensemble_ece = expected_calibration_error_from_predictions(bayes_predictions,
                                                                                     test_predictions["Target"],
                                                                                     _config["nbins"])
                    data_dict["bayes_ensemble_expected_calibration_error"] = float(bayes_ensemble_ece)
                    logger.info(f"Bayes ensemble expected calibration error: {bayes_ensemble_ece} \n")
                else:
                    bayes_ensemble_log_likelihood = log_likelihood(bayes_ensemble, test_dataset,
                                                                   _config["test_batch_size"])
                    data_dict["bayes_ensemble_log_likelihood"] = bayes_ensemble_log_likelihood
                    logger.info(f"Bayes ensemble log likelihood: {bayes_ensemble_log_likelihood}")
                    bayes_ensemble_accuracy = accuracy(bayes_ensemble, test_dataset,
                                                       _config["test_batch_size"])
                    data_dict["bayes_ensemble_accuracy"] = bayes_ensemble_accuracy
                    logger.info(f"Bayes ensemble accuracy: {bayes_ensemble_accuracy}")
                    bayes_ensemble_ece = expected_calibration_error(bayes_ensemble, test_dataset, _config["nbins"],
                                                                    _config["test_batch_size"])
                    data_dict["bayes_ensemble_expected_calibration_error"] = float(bayes_ensemble_ece)
                    logger.info(f"Bayes ensemble expected calibration error: {bayes_ensemble_ece} \n")

                if ("random", len(models_used)) in _run.info:
                    _run.info[("random", len(models_used))] = dict_cat(_run.info[("random", len(models_used))], data_dict)
                else:
                    _run.info[("random", len(models_used))] = data_dict