"""
Example model for use in ensemble for MNIST
Shallow CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.cifar10 as dataset_module


class MLP3(nn.Module):
    """
    MLP with 8 layers and some dropout
    """
    def __init__(self, trained=True):
        super().__init__()
        self.dropout = nn.Dropout1d()
        self.fc1 = nn.Linear(3072, 2400)
        self.fc2 = nn.Linear(2400, 1200)
        self.fc3 = nn.Linear(1200, 800)
        self.fc4 = nn.Linear(800, 400)
        self.fc5 = nn.Linear(400, 200)
        self.fc6 = nn.Linear(200, 100)
        self.fc7 = nn.Linear(100, 50)
        self.fc8 = nn.Linear(50, 10)



        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\mlp3.pth'))

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        x = F.selu(self.fc5(x))
        x = self.dropout(x)
        x = F.gelu(self.fc6(x))
        x = F.gelu(self.fc7(x))
        x = F.softmax(self.fc8(x))
        return x



if __name__ == "__main__":
    model = MLP3(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    torch_train(model, train_dataset, test_dataset, "mlp3", device=torch.device("cuda:0"), learning_rate=0.01, momentum=0.9, batch_size_train=64, gamma=0.9, n_epochs=20)

