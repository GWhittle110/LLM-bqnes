# MNIST Dataset

from xautodl.datasets.DownsampledImageNet import ImageNet16
import torchvision
import os
import git

abs_path = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir,
                        "experiments\\datasets\\data\\ImageNet16")

train_dataset = ImageNet16(abs_path, train=True,
                           transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ]),
                           use_num_of_class_only=120)

test_dataset = ImageNet16(abs_path, train=False,
                          transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                          ]),
                          use_num_of_class_only=120)
