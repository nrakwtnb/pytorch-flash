
#from .dev import forward_wrap
from utils import forward_wrap

import torch.nn as nn
import torch.nn.functional as F

channels = [1,16,32]
units = [96]
kernel_size = 3
width = 28
height = 28
num_classes = 10
final_width = ((width - kernel_size + 1) // 2 - kernel_size + 1) // 2
final_height = ((height - kernel_size + 1) // 2 - kernel_size + 1) // 2
units = [ final_width * final_height * channels[-1] ] + units
print(final_width, final_height)



class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(units[0], units[1])
        self.fc2 = nn.Linear(units[1], num_classes)

    @forward_wrap
    def forward(self, inputs):
        x = inputs["x"]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, units[0])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return { "y" : F.log_softmax(x, dim=-1) }


# MNIST data loader
# taken from pytorch-ignite mnist examples

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

def get_data_loaders(train_batch_size, val_batch_size, mnist_path, train_dataset_size=None, val_dataset_size=None, download=False):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(download=download, root=mnist_path, transform=data_transform, train=True)
    val_dataset = MNIST(download=download, root=mnist_path, transform=data_transform, train=False)

    import numpy as np
    if isinstance(train_dataset_size, int):
        train_dataset.data = train_dataset.data[:train_dataset_size]
        train_dataset.targets = train_dataset.targets[:train_dataset_size]
    elif isinstance(train_dataset_size, list) or isinstance(train_dataset_size, np.ndarray):
        train_dataset.data = train_dataset.data[train_dataset_size]
        train_dataset.targets = train_dataset.targets[train_dataset_size]
    if isinstance(val_dataset_size, int):
        val_dataset.data = val_dataset.data[:val_dataset_size]
        val_dataset.targets = val_dataset.targets[:val_dataset_size]
    elif isinstance(val_dataset_size, list) or isinstance(val_dataset_size, np.ndarray):
        val_dataset.data = val_dataset.data[val_dataset_size]
        val_dataset.targets = val_dataset.targets[val_dataset_size]
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def get_state(manager, engine_name='trainer'):
    return manager.config['objects']['engine'][engine_name].state
