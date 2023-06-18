import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import os
from PIL import Image

from source import transform


class FaceDataset(Dataset):
    def __init__(self, image_dir: str, transform=transform):
        self.root_dir = image_dir
        self.transform = transform
        self.image_paths: list = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)

        label = self._get_label_old(image_path)

        return image, label

    def _get_image_paths(self) -> list:
        data = []
        categories = os.listdir(self.root_dir)
        for image_class in categories:

            full_path = os.path.join(self.root_dir, image_class)
            images = os.listdir(full_path)
            for image_name in images:
                im_path = os.path.join(full_path, image_name)
                data.append(im_path)
        return data

    @staticmethod
    def _get_label(image_path):
        tens = torch.load('middle_tensor_dict.pth')
        label = os.path.basename(os.path.dirname(image_path))
        label = torch.squeeze(tens[label])
        return label

    @staticmethod
    def _get_label_old(image_path):
        label = os.path.basename(os.path.dirname(image_path))
        label = torch.tensor(int(label))
        return label


dataset_dir = 'dataset'
train_set = FaceDataset(os.path.join(dataset_dir, 'train'), transform=transform)
train_loader = DataLoader(train_set, batch_size=15, shuffle=True)

# count = 0
# for im, lab in train_set:
#     # print(f'len lab: {lab.shape} | Lab: {lab}')
#     # print(f'Len tensor: {im.shape} | tensor: {im}')
#     count += 1
#     if count == 1000: break


test_set = FaceDataset(os.path.join(dataset_dir, 'test'), transform=transform)
test_loader = DataLoader(train_set, batch_size=15, shuffle=True)
