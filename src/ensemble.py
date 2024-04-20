"""
Classes to handle ensemble of models
"""

import pandas as pd
import torch
from src.bayes_quad.quadrature import *
from src.utils.rebalanceProbabilities import rebalance_probabilities


class Ensemble(torch.nn.Module):
    """
    Standard ensemble deriving from Bayesian Quadrature
    """

    def __init__(self, models, integrand: IntegrandModel):
        """
        :param models: Constituent models
        :param integrand: Integrand model from which the ensemble is derived
        """
        super().__init__()
        self.models = models
        self.quad_weights = integrand.quad_weights.astype(np.float64)
        self.likelihoods = integrand.surrogate.y.astype(np.float64)
        self.evidence = integrand.evidence
        self.weights = self.quad_weights * self.likelihoods / self.evidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        >>> x1 = torch.tensor(1.)
        >>> x2 = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        >>> print(ensemble(x1))
        >>> print(ensemble(x2))
        """
        member_predictions = torch.stack([model(x) for model in self.models], -1)
        return rebalance_probabilities(member_predictions @ self.weights)

    def forward_from_predictions(self, predictions_df: pd.DataFrame) -> torch.Tensor:
        """
        Calculate ensemble predictions from predictions of constituent models
        :param predictions_df: Dataset containing predictions of all possible ensemble models, keyed on model name
        :return: tensor of model predictions
        """
        member_predictions = torch.stack([torch.tensor(predictions_df[type(model).__name__])
                                          for model in self.models], -1)
        return rebalance_probabilities(member_predictions @ self.weights)


class SqEnsemble(Ensemble):
    """
    Ensemble deriving from square root warped Bayesian Quadrature
    """

    def __init__(self, models, integrand: SqIntegrandModel):
        """
        :param models: Constituent models
        :param integrand: Integrand from which ensemble is derived
        """
        super().__init__(models, integrand)
        self.weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        >>> x1 = torch.tensor(1.)
        >>> x2 = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        >>> print(ensemble(x1))
        >>> print(ensemble(x2))

        """
        member_predictions = torch.stack([model(x) for model in self.models], -1)
        numerator_integrand = member_predictions * self.likelihoods
        epsilon = 0.8 * torch.min(numerator_integrand, dim=-1).values
        z = torch.sqrt(2 * (numerator_integrand - epsilon.unsqueeze(-1)))
        ensemble_prediction = rebalance_probabilities((epsilon +
            0.5 * torch.einsum('...i, ij, ...j -> ...', z, torch.tensor(self.quad_weights), z)) / self.evidence)
        return ensemble_prediction

    def forward_from_predictions(self, predictions_df: pd.DataFrame) -> torch.Tensor:
        """
        Calculate ensemble predictions from predictions of constituent models
        :param predictions_df: Dataset containing predictions of all possible ensemble models, keyed on model name
        :return: tensor of model predictions
        """
        member_predictions = torch.stack([torch.tensor(predictions_df[type(model).__name__])
                                          for model in self.models], -1)
        numerator_integrand = member_predictions * self.likelihoods
        epsilon = 0.8 * torch.min(numerator_integrand, dim=-1).values
        z = torch.sqrt(2 * (numerator_integrand - epsilon.unsqueeze(-1)))
        ensemble_prediction = rebalance_probabilities((epsilon +
             0.5 * torch.einsum('...i, ij, ...j -> ...', z, torch.tensor(self.quad_weights), z)) / self.evidence)
        return ensemble_prediction


class LinSqEnsemble(Ensemble):
    """
    Ensemble deriving from linearisation of square root warped Bayesian Quadrature
    """

    def __init__(self, models, integrand: SqIntegrandModel):
        """
        :param models: Constituent models
        :param integrand: Integrand from which ensemble is derived
        """
        super().__init__(models, integrand)
        self.offset = (1 - self.quad_weights.sum()) / self.evidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        >>> x1 = torch.tensor(1.)
        >>> x2 = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        >>> print(ensemble(x1))
        >>> print(ensemble(x2))

        """
        member_predictions = torch.stack([model(x) for model in self.models], -1)
        numerator_integrand = member_predictions * self.likelihoods
        epsilon = 0.8 * torch.min(numerator_integrand, dim=-1).values
        ensemble_prediction = rebalance_probabilities(epsilon * self.offset + member_predictions @ self.weights)
        return ensemble_prediction

    def forward_from_predictions(self, predictions_df: pd.DataFrame) -> torch.Tensor:
        """
        Calculate ensemble predictions from predictions of constituent models
        :param predictions_df: Dataset containing predictions of all possible ensemble models, keyed on model name
        :return: tensor of model predictions
        """
        member_predictions = torch.stack([torch.tensor(predictions_df[type(model).__name__])
                                          for model in self.models], -1)
        numerator_integrand = member_predictions * self.likelihoods
        epsilon = 0.8 * torch.min(numerator_integrand, dim=-1).values
        ensemble_prediction = rebalance_probabilities(epsilon * self.offset + member_predictions @ self.weights)
        return ensemble_prediction


class DiagSqEnsemble(Ensemble):
    """
    Ensemble using diagonal of quad weights from square root warped Bayesian Quadrature
    """

    def __init__(self, models, integrand: SqIntegrandModel):
        """
        :param models: Constituent models
        :param integrand: Integrand from which the ensemble is derived. In this case the diag sq ensemble is initialised
        from an equivalent SqIntegrandModel, not a DiagSqIntegrandModel
        """
        torch.nn.Module.__init__(self)
        self.models = models
        self.quad_weights = integrand.quad_weights.diagonal() / integrand.quad_weights.trace()
        self.likelihoods = integrand.surrogate.y
        self.evidence = self.quad_weights @ self.likelihoods
        self.weights = self.quad_weights * self.likelihoods / self.evidence


class UniformEnsemble(Ensemble):
    """
    Ensemble using uniform weights
    """

    def __init__(self, models):
        """
        Ensemble using uniform weights
        :param models: Constituent models
        """
        torch.nn.Module.__init__(self)
        self.models = models
        self.weights = np.ones(len(models)) / len(models)


class BayesEnsemble(Ensemble):
    """
    Ensemble using likelihoods as weights
    """

    def __init__(self, models, integrand):
        super().__init__(models, integrand)
        self.evidence = np.sum(self.likelihoods)
        self.weights = self.likelihoods / self.evidence
