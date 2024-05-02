"""
LGBM model for CIFAR-10 dataset
"""

import torch
import torch.nn as nn
import lightgbm as lgbm
from experiments.trainers.SKTrain import sk_train
import experiments.datasets.cifar10 as dataset_module
import joblib
from pickle import load

class LGBM(nn.Module):
    """
    LightGBM model for CIFAR-10, interface with pytorch
    """
    def __init__(self, trained=True, *args, **kwargs):
        super().__init__()
        self.model = lgbm.LGBMClassifier(*args, **kwargs)
        if trained:
            self.model = joblib.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\lgbm.pkl')

    def forward(self, x):
        x = x.reshape(-1, 3072)
        return torch.tensor(self.model.predict_proba(x.cpu().numpy()))

    def fit(self, X, Y):
        X = X.reshape(-1, 3072)
        self.model.fit(X.numpy(), Y.numpy())

    def predict_proba(self, x):
        x = x.reshape(-1, 3072)
        return torch.tensor(self.model.predict_proba(x.numpy()))

    def save_model(self):
        joblib.dump(self.model, './states/lgbm.pkl')


if __name__ == "__main__":
    model = LGBM(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    sk_train(model, train_dataset, test_dataset)
