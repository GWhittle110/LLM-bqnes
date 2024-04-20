# Logistic regression dataset

import pandas as pd
from torch.utils.data import Dataset
import torch


class LogisticDataset(Dataset):

    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.classes = {0, 1}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


D_train = pd.read_csv("C:\\Users\\gwhit\\PycharmProjects\\4YP\\experiments\\datasets\\csvs\\logistic_train.csv")
X_train = D_train.values[:, :-1]
Y_train = D_train.values[:, -1]

D_test = pd.read_csv("C:\\Users\\gwhit\\PycharmProjects\\4YP\\experiments\\datasets\\csvs\\logistic_test.csv")
X_test = D_test.values[:, :-1]
Y_test = D_test.values[:, -1]

train_dataset = LogisticDataset(X_train, Y_train)
test_dataset = LogisticDataset(X_test, Y_test)