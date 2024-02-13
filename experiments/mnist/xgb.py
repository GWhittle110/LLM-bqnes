"""
Example model for use in ensemble for MNIST
XGBoost
"""

import torch
import xgboost as xgb
from mnistEnsembleExample.xgbTrain import xgbTrain


class XGB:
    """
    XGBoost model for MNIST, interface with pytorch
    """
    def __init__(self, trained=True, *args, **kwargs):
        self.model = xgb.XGBClassifier(*args, **kwargs)
        if trained:
            self.model.load_model("C:/Users/gwhit/PycharmProjects/4YP/mnistEnsembleExample/states/xgb.json")

    def __call__(self, x):
        x = x.reshape(-1, 784)
        return torch.tensor(self.model.predict_proba(x.numpy()))

    def fit(self, X, Y):
        X = X.reshape(-1, 784)
        self.model.fit(X.numpy(), Y.numpy())

    def predict_proba(self, x):
        x = x.reshape(-1, 784)
        return torch.tensor(self.model.predict_proba(x.numpy()))

    def save_model(self):
        self.model.save_model("./states/xgb.json")


if __name__ == "__main__":
    model = XGB(trained=False)
    xgbTrain(model)
