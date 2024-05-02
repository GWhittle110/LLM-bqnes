# MNIST Dataset

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                             transform=transform)

test_dataset = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                            transform=transform)
