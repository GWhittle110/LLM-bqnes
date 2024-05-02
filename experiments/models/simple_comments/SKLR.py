"""
Logistic regression model
"""

import torch
from torch import nn
import sklearn.linear_model as lm
from experiments.trainers.SKTrain import sk_train
import experiments.datasets.logistic as dataset_module
from pickle import dump, load


train_dataset = getattr(dataset_module, "train_dataset")
test_dataset = getattr(dataset_module, "test_dataset")


class SKLR(nn.Module):
    """
    Logistic regression model implemented in Scikit-Learn
    """

    def __init__(self, trained=True, *args, **kwargs):
        super().__init__()
        if trained:
            with open('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\simple\\states\\sklr.pkl', 'rb') as f:
                self.model = load(f)
        else:
            self.model = lm.LogisticRegression(*args, **kwargs)

    def forward(self, x):
        x = x.reshape(-1, 2)
        return torch.tensor(self.model.predict_proba(x.cpu().numpy()))

    def fit(self, X, Y):
        X = X.reshape(-1, 2)
        self.model.fit(X.numpy(), Y.numpy())

    def predict_proba(self, x):
        x = x.reshape(-1, 2)
        return torch.tensor(self.model.predict_proba(x.numpy()))

    def save_model(self):
        with open('./states/sklr.pkl', 'wb+') as f:
            dump(self.model, f)


if __name__ == "__main__":
    model = SKLR(trained=False)
    sk_train(model, train_dataset, test_dataset)