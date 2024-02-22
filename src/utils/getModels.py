"""
Function to extract models in order used by integrand
"""

from src.bayes_quad.quadrature import IntegrandModel
from src.LLMSearchSpace.searchSpace import SearchSpace


def get_models(integrand: IntegrandModel, search_space: SearchSpace) -> list:
    """
    Extract models in order used by integrand
    :param integrand: Integrand model
    :param search_space: Search space containing models
    :return: List containing models in order
    """
    return [search_space.models_dict[tuple(coord)] for coord in integrand.surrogate.x]
