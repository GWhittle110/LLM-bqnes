"""
Example model for use in ensemble for MNIST
Shallow CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.cifar10 as dataset_module


class MLP1(nn.Module):
    """
    MLP with 6 layers and no dropout
    """
    def __init__(self, trained=True):
        super().__init__()
        self.fc1 = nn.Linear(3072, 1600)
        self.fc2 = nn.Linear(1600, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 10)


        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\mlp1.pth'))

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = F.softmax(x)
        return x



if __name__ == "__main__":
    model = MLP1(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    torch_train(model, train_dataset, test_dataset, "mlp1", device=torch.device("cuda:0"), learning_rate=0.01, momentum=0.9, batch_size_train=64, gamma=0.9, n_epochs=10)

