import pprint

import cv2
import numpy as np
import torch.onnx
import torchvision.models
from sklearn.neighbors import KNeighborsClassifier
import os
from torchsummary import summary
import torch
from tqdm.auto import tqdm

from datasets import train_loader, test_loader
from source import num_classes

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


def save_model(model_name='model.pth'):
    dir_to_save = os.path.join('models', model_name)
    torch.save(model.state_dict(), dir_to_save)


def load_model(model_name='model.pth'):
    dir_to_load = os.path.join('models', model_name)
    model.load_state_dict(torch.load(dir_to_load))


class ExtraLayers(nn.Module):
    def __init__(self):
        super(ExtraLayers, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT).to(device)

extra_layers = ExtraLayers()

model.fc = extra_layers.to(device)

# Определите гиперпараметры
learning_rate = 0.0001
weight_decay = 0.0001
num_epochs = 50

# Определите функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Подготовьте данные для обучения и тестирования
best_loss = float('inf')

# # Цикл обучения
# for epoch in range(num_epochs):
#     loss_val = 0.0
#
#     model.train()
#     with tqdm(total=test_loader.__len__(), position=0) as progress:
#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             # Обнулите градиенты
#             optimizer.zero_grad()
#
#             # Передайте данные через модель
#             outputs = model(images)
#             # Вычислите потери
#             loss = criterion(outputs, labels)
#
#             if loss < best_loss:
#                 best_loss = loss
#                 save_model(model_name='model_classifier_resnet_50_v2.pth')
#
#             # Обновите веса
#             loss.backward()
#
#             loss_item = loss.item()
#             loss_val += loss_item
#
#             optimizer.step()
#             progress.set_description(f"Epoch: {epoch} | Training loss(iter): {str(loss_item)[:6]}")
#             progress.update()
#         progress.set_description(f"Epoch: {epoch} | Training loss(total): {str(loss_val / len(train_loader))[:6]}")

load_model(model_name='model_classifier_resnet_50_v2.pth')
# # Цикл тестирования
# correct = 0
# total = 0
# test_loss = 0.0
# with torch.no_grad():
#     model.eval()
#     for data in test_loader:
#         image, label = data[0].to(device), data[1].to(device)
#         output = model(image)
#         test_loss += criterion(output, label).item()
#         _, predicted = torch.max(output.data, 1)
#         total += label.size(0)
#         correct += (predicted == label).sum().item()
#
# print(f"Test Loss: {test_loss / total} | Accuracy: {100 * correct / total}")

tens_to_save = {f'{i}': [] for i in range(num_classes)}
middle_tensors = {}

with torch.no_grad():
    model.eval()
    for images, labels in train_loader:
        for im, lab in zip(images.to(device), labels.to(device)):
            im = im.unsqueeze(0)
            out = model(im)
            tens_to_save[f'{lab}'].append(out)
    torch.save(tens_to_save, 'tensor_dict_v3.pth')


for key in tens_to_save.keys():
    temp = 0
    for tensor in tens_to_save[key]:
        temp += tensor
    middle = temp / len(tens_to_save[key])
    middle_tensors[key] = middle
pprint.pprint(middle_tensors)
torch.save(middle_tensors, 'middle_tensor_dict_v3.pth')

