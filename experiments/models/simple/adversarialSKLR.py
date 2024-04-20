"""
Adversarial logistic regression model
"""

import torch
from torch import nn
import torch.nn.functional as F
import sklearn.linear_model as lm
from experiments.trainers.SKTrain import SKTrain
import experiments.datasets.logistic as dataset_module
from pickle import dump, load


train_dataset = getattr(dataset_module, "train_dataset")
test_dataset = getattr(dataset_module, "test_dataset")


class AntiSKLR(nn.Module):

    def __init__(self, trained=True, *args, **kwargs):
        super().__init__()
        if trained:
            with open('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\simple\\states\\adversarialsklr.pkl', 'rb') as f:
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
        return F.softmax(-torch.log(torch.tensor(self.model.predict_proba(x.numpy()))))

    def save_model(self):
        with open('./states/adversarialsklr.pkl', 'wb+') as f:
            dump(self.model, f)


if __name__ == "__main__":
    model = AntiSKLR(trained=False)
    SKTrain(model, train_dataset, test_dataset)