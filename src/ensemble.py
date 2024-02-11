"""
Classes to handle ensemble of models
"""

import numpy as np


class Ensemble:
    """
    Standard ensemble deriving from Bayesian Quadrature
    """

    def __init__(self, members, quad_weights: np.ndarray, likelihoods: np.ndarray, evidence: float):
        """
        :param members: Constituent models
        :param quad_weights: Quadrature scheme weights
        :param likelihoods: Model training likelihoods
        :param evidence: Ensemble evidence
        """
        self.members = members
        self.quad_weights = quad_weights
        self.likelihoods = likelihoods
        self.evidence = evidence

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call ensemble
        :param x: Input value
        :return: Ensemble prediction
        Example:
        >>> model1 = lambda x: x
        >>> model2 = lambda x: 2 * x
        >>> models = [model1, model2]
        >>> quad_weights = np.array([2.,1.])
        >>> likelihoods = np.array([0.5,0.5])
        >>> evidence = quad_weights @ likelihoods
        >>> ensemble = Ensemble(models, quad_weights, likelihoods, evidence)
        >>> x1 = np.array(1)
        >>> x2 = np.array([[1,2,3],[4,5,6]])
        >>> print(ensemble(x1))
        >>> print(ensemble(x2))
        """
        member_predictions = np.stack([model(x) for model in self.members], -1)
        weights = self.quad_weights * self.likelihoods / self.evidence
        return member_predictions @ weights


class SqEnsemble(Ensemble):
    """
    Ensemble deriving from square root warped Bayesian Quadrature
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call ensemble
        :param x: Input value
        :return: Ensemble prediction
        >>> model1 = lambda x: x
        >>> model2 = lambda x: 2 * x
        >>> models = [model1, model2]
        >>> quad_weights = np.array([[2.,1.],[1.,2.]])
        >>> likelihoods = np.array([0.5,0.5])
        >>> epsilon = 0.8 * np.min(likelihoods)
        >>> z = np.sqrt(2 * (likelihoods - epsilon))
        >>> evidence = epsilon + 0.5 * z @ quad_weights @ z
        >>> ensemble = SqEnsemble(models, quad_weights, likelihoods, evidence)
        >>> x1 = np.array(1)
        >>> x2 = np.array([[1,2,3],[4,5,6]])
        >>> print(ensemble(x1))
        >>> print(ensemble(x2))

        """
        member_predictions = np.stack([model(x) for model in self.members], -1)
        numerator_integrand = member_predictions * self.likelihoods
        epsilon = 0.8 * np.min(numerator_integrand, axis=-1)
        z = np.sqrt(2 * (numerator_integrand - np.expand_dims(epsilon, axis=-1)))
        breakpoint()
        ensemble_prediction = ((epsilon +
                               0.5 * np.einsum('...i, ij, ...j -> ...', z, self.quad_weights, z))
                               / self.evidence)
        return ensemble_prediction
