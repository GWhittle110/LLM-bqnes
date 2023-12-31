"""
Isolated trainer for xgboost models
"""

import torch
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import torchvision


def xgbTrain(model):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=60000, shuffle=False)

    _, (training_data, training_targets) = next(enumerate(test_loader))

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=10000, shuffle=False)

    _, (testing_data, testing_targets) = next(enumerate(test_loader))

    model.fit(training_data, training_targets)
    model.save_model()
    y_probs = model.predict_proba(testing_data)
    y_preds = np.array([np.argmax(prob) for prob in y_probs])
    loss = log_loss(testing_targets.numpy(), y_probs.numpy(), normalize=True)
    acc = accuracy_score(testing_targets.numpy(), y_preds)
    print(f'Testing loss: {loss}')
    print(f'Testing accuracy: {acc}')

