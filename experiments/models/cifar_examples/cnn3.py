"""
Example model for use in ensemble for MNIST
Shallow CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.trainers.torchTrain import torch_train
import experiments.datasets.cifar10 as dataset_module


class CNN3(nn.Module):
    """
    CNN with 4 convolutional layers, some dropout and 5 fully connected layers
    """
    def __init__(self, trained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.dropout = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 24, 3)
        self.conv4 = nn.Conv2d(24, 32, 3)
        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)


        if trained:
            self.load_state_dict(torch.load('C:\\Users\\gwhit\PycharmProjects\\4YP\experiments\\models\\cifar_basic\\states\\cnn3.pth'))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x))
        return x



if __name__ == "__main__":
    model = CNN3(trained=False)
    train_dataset = getattr(dataset_module, "train_dataset")
    test_dataset = getattr(dataset_module, "test_dataset")
    torch_train(model, train_dataset, test_dataset, "cnn3", device=torch.device("cuda:0"), learning_rate=0.02, momentum=0.9, batch_size_train=64, gamma=0.9, n_epochs=20)

