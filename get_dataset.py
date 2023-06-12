import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        # image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Возвращаем изображение и метку класса (в данном случае, идентификатор человека)
        label = self._get_label_from_path(image_path)
        return image, label

    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _get_label_from_path(self, image_path):
        # Извлекаем метку класса из пути к изображению.
        # В данном примере предполагается, что имена папок являются метками классов (идентификаторами человека).
        # Может потребоваться настройка в соответствии с вашей структурой данных.
        label = os.path.basename(os.path.dirname(image_path))
        return label
