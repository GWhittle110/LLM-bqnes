"""
Example model for use in ensemble for MNIST
XGBoost
"""

import torch
import torch.nn as nn
import xgboost as xgb
from sandbox.mnistEnsembleExample import xgbTrain


class XGB(nn.Module):
    """
    XGBoost model for MNIST, interface with pytorch
    """
    def __init__(self, trained=True, *args, **kwargs):
        super().__init__()
        self.model = xgb.XGBClassifier(*args, **kwargs)
        if trained:
            self.model.load_model('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\mnist_basic\\states\\xgb.json')

    def forward(self, x):
        x = x.reshape(-1, 784)
        return torch.tensor(self.model.predict_proba(x.cpu().numpy()))

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
