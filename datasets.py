from torch.utils.data import DataLoader
from torchvision import datasets

import os

from source import transform

dataset_dir = 'dataset'
train = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
trainset = DataLoader(train, batch_size=15, shuffle=True)

test = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
testset = DataLoader(train, batch_size=15, shuffle=True)
