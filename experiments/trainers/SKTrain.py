"""
Isolated trainer for sklearn models
"""

import torch
import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def SKTrain(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    _, (training_data, training_targets) = next(enumerate(train_loader))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    _, (testing_data, testing_targets) = next(enumerate(test_loader))

    model.fit(training_data, training_targets)
    model.save_model()
    y_probs = model.predict_proba(testing_data)
    y_preds = np.array([np.argmax(prob) for prob in y_probs])
    loss = log_loss(testing_targets.numpy(), y_probs.numpy(), normalize=True)
    acc = accuracy_score(testing_targets.numpy(), y_preds)
    print(f'Testing loss: {loss}')
    print(f'Testing accuracy: {acc}')

