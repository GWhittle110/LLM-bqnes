# MNIST Dataset

import torchvision

train_dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

test_dataset = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
