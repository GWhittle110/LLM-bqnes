"""
Function to load models from specified folder
"""

import inspect
import re
from importlib import import_module
import os
import git


def load_models(path: str) -> list:
    """
    Load models at location specified by path
    :param path: Path to directory containing model source code, from repository root
    :return: list of models
    """
    package_path = re.sub('/|[\\\\]', '.', path)
    abs_path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir, path)
    module_names = [file.rsplit('.', 1)[0] for file in os.listdir(abs_path)
                    if '__' not in file and os.path.isfile(os.path.join(abs_path, file))]
    modules = [import_module(f'{package_path}.{module_name}') for module_name in module_names]

    def eval_mode(model):
        try:
            return model.eval()
        except AttributeError:
            return model

    return [eval_mode(model()) for module in modules for _, model in inspect.getmembers(module, inspect.isclass)]
