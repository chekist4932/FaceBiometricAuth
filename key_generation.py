import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import os
from PIL import Image
from tqdm.auto import tqdm

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

        label = self._get_label(image_path)

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


class BiometricsToAccessCodeTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(BiometricsToAccessCodeTransformer, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * (input_size // 8) * (input_size // 8), 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = 'dataset'
train_set = FaceDataset(os.path.join(dataset_dir, 'train'), transform=transform)
train_loader = DataLoader(train_set, batch_size=15, shuffle=True)

# count = 0
# for im, lab in train_set:
#     print(lab)
#     target = torch.argmax(lab, dim=1)  # Преобразование в одномерный тензор
#     print(target)  # Вывод одн
#
#     # print(f'len lab: {lab.shape} | Lab: {lab}')
#     # print(f'Len tensor: {im.shape} | tensor: {im}')
#     count += 1
#     if count == 10: break
#
# input('Stop')


test_set = FaceDataset(os.path.join(dataset_dir, 'test'), transform=transform)
test_loader = DataLoader(train_set, batch_size=15, shuffle=True)

input_size = 224  # Размер входного изображения (64x64)
output_size = 32  # Количество классов/кодов доступа

# Создание экземпляра модели
model = BiometricsToAccessCodeTransformer(input_size, output_size)
model.to(device)

# Определение функции потерь
# criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.MSELoss().to(device)

# Определение оптимизатора
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

use_amp = False
torch.backends.cudnn.benchmark = True
# Цикл обучения
# num_epochs = 200
# try:
#     for epoch in range(num_epochs):
#         loss_val = 0.0
#         model.train()  # Перевод модели в режим обучения
#         with tqdm(total=test_loader.__len__(), position=0) as progress:
#             for images, labels in train_loader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 optimizer.zero_grad()  # Обнуление градиентов
#
#                 outputs = model(images)  # Прямой проход через модель
#                 loss = criterion(outputs, labels)  # Вычисление функции потерь
#
#                 loss.backward()
#                 loss_item = loss.item()
#                 loss_val += loss_item
#
#                 optimizer.step()
#                 progress.set_description(f"Epoch: {epoch} | Training loss(iter): {str(loss_item)[:6]}")
#                 progress.update()
#             progress.set_description(f"Epoch: {epoch} | Training loss(total): {str(loss_val / len(train_loader))[:6]}")
# except KeyboardInterrupt:
#     input("Save?")
#     dir_to_save = os.path.join('models', 'model_test_Adam_200.pth')
#     torch.save(model.state_dict(), dir_to_save)
# else:
#     dir_to_save = os.path.join('models', 'model_test_Adam_200.pth')
#     torch.save(model.state_dict(), dir_to_save)
dir_to_load = os.path.join('models', 'model_test_Adam_200.pth')
model.load_state_dict(torch.load(dir_to_load))

test_loss = 0.0
correct = 0

with torch.no_grad():
    model.eval()
    total = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # for im, lab in zip(images, labels):
        #     im = im.unsqueeze(0)
        #     outputs = model(im)
        #     print(torch.mean(outputs))
        #     print(torch.mean(lab))
        #     print(f"{torch.mean(torch.isclose(outputs, lab, rtol=0.1, atol=0.1).float())}")
        # input("Stop")
        outputs = model(images)

        test_loss += criterion(outputs, labels).item()
        correct += torch.mean(torch.isclose(outputs, labels, rtol=0.1, atol=0.1).float())
        # correct += (outputs == labels).sum().item()

test_loss /= len(test_loader)
accuracy = 100 * correct / len(test_loader)

# Вывод результатов после каждой эпохи
print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
