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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT).to(device)

extra_layers = nn.Sequential(
    nn.Linear(in_features=512 * models.resnet.BasicBlock.expansion, out_features=128),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=128, out_features=num_classes),

)

model.fc = extra_layers.to(device)

# Определите гиперпараметры
learning_rate = 0.0001
num_epochs = 10

# Определите функцию потерь и оптимизатор
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Подготовьте данные для обучения и тестирования
best_loss = float('inf')

# Цикл обучения
for epoch in range(num_epochs):
    loss_val = 0.0

    model.train()
    with tqdm(total=test_loader.__len__(), position=0) as progress:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Обнулите градиенты
            optimizer.zero_grad()

            # Передайте данные через модель
            outputs = model(images)
            # Вычислите потери
            loss = criterion(outputs, labels)

            if loss < best_loss:
                best_loss = loss
                save_model(model_name='model_classifier_resnet_MSE.pth')

            # Обновите веса
            loss.backward()

            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()
            progress.set_description(f"Epoch: {epoch} | Training loss(iter): {str(loss_item)[:6]}")
            progress.update()
        progress.set_description(f"Epoch: {epoch} | Training loss(total): {str(loss_val / len(train_loader))[:6]}")

    # Печать информации о потерях после каждой эпохи
load_model(model_name='model_classifier_resnet_MSE.pth')
# Цикл тестирования
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in test_loader:
        image, label = data[0].to(device), data[1].to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f"Accuracy: {100 * correct / total}")

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(16 * 112 * 112, 256)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         threshold = 0.5
#         x = (x >= threshold).float()
#         return x
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Создание экземпляра модели
# model = ConvNet().to(device)
#
# criterion = nn.MSELoss().to(device)  # Функция потерь
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Оптимизатор
# num_epochs = 10
# # Цикл обучения
# for epoch in range(num_epochs):
#     # Обучение на тренировочных данных
#     with tqdm(total=train_loader.__len__(), position=0) as progress:
#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()  # Обнуление градиентов
#             outputs = model(images)  # Прямой проход
#             loss = criterion(outputs, labels)  # Вычисление потерь
#             loss.backward()  # Обратное распространение
#             optimizer.step()  # Обновление параметров
#             progress.update()
#         # Вывод информации о процессе обучения
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# input(outputs)
# input(labels)
# model.eval()  # Перевод модели в режим тестирования
# test_loss = 0.0
# total_samples = 0
# correct_samples = 0
#
# with torch.no_grad():
#     for images, labels in test_loader:
#         # Передача данных на GPU (если доступно)
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Вычисление предсказаний
#         outputs = model(images)
#
#         # Вычисление функции потерь
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()
#
#         # Подсчет общего числа образцов и правильных предсказаний
#         total_samples += labels.size(0)
#         predicted_labels = torch.round(outputs)
#         # correct_samples += (predicted_labels == labels).sum().item()
#         correct_samples += (predicted_labels == labels)
#
# # Вывод результатов тестирования
# test_loss /= len(test_loader)
# accuracy = 100.0 * correct_samples / total_samples
# print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')

# tensor = torch.randint(low=0, high=2, size=(256,), dtype=torch.float32)
# print(tensor)
#

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
# # model.to(device)
# extra_layers = nn.Sequential(
#     nn.Linear(in_features=512 * models.resnet.BasicBlock.expansion, out_features=256),
#     nn.ReLU(inplace=True),
#     nn.Linear(in_features=256, out_features=38),
#     nn.Sigmoid()
# )
#
# # Добавляем пороговую функцию для бинаризации выхода
#
# # Заменяем последние слои модели на нашу последовательность слоев
# model.fc = extra_layers
#
# # Пример использования модели
# input_data = torch.randn(1, 3, 224, 224)  # Входные данные размером 1x3x224x224
# with torch.no_grad():
#     model.eval()
#     output = model(input_data)  # Вычисление выхода модели
#     threshold = 0.5
#     output = (output >= threshold).float()
#     print(output)

# model = torchvision.models.resnet34().to(device='cuda')
# summary(model, (3, 224, 224))

# Загрузка изображения
# photo_dir = 'dataset/train/0/IMG_4970.jpg'
# img = cv2.imread(photo_dir)
#
# # Изменение размера изображения
# img = cv2.resize(img, (800, 600))
#
# #Извлечение лицевого ключа с помощью обученной нейронной сети
#
# # Загрузка нейронной сети
# dir_ = 'models/openface.nn4.small2.v1 (1).t7'
# model = torch.load(dir_)

# model = cv2.dnn.readNetFromTorch(dir_)
#
# # Извлечение лицевого ключа из изображения
# blob = cv2.dnn.blobFromImage(img, 1, (96, 96), (0, 0, 0), swapRB=True, crop=False)
# model.setInput(blob)
# face = model.forward()
# print(face)
# print(face.shape)
# Обучение классификатора KNN

# Создание словаря, где ключ - метка, значение - список лицевых ключей
# database = {}
#
# # Цикл по папкам с изображениями каждого класса
# for class_name in os.listdir(r'D:\100'):
#
#     # Создание списка лицевых ключей для текущего класса
#     class_keys = []
#
#     # Цикл по файлам изображений в каждой папке класса
#     for file_name in os.listdir(os.path.join(r'D:\100', class_name)):
#         # Загрузка изображения и изменение его размера
#         img = cv2.imread(os.path.join(r'D:\100', class_name, file_name))
#         img = cv2.resize(img, (800, 600))
#
#         # Извлечение лицевого ключа из изображения
#         blob = cv2.dnn.blobFromImage(img, 1, (96, 96), (0, 0, 0), swapRB=True, crop=False)
#         model.setInput(blob)
#         key = model.forward()
#
#         # Добавление ключа в список текущего класса
#         class_keys.append(key)
#
#     # Добавление списка ключей текущего класса в базу данных
#     database[class_name] = class_keys
#
# # Сохранение базы данных в файл "database.npy"
# np.save('database.npy', database)
# # Загрузка базы данных с лицами и метками
# database = np.load('database.npy', allow_pickle=True).item()
#
# # Составление списка с лицевыми ключами и метками из базы данных
# keys = []
# labels = []
# for key, value in database.items():
#     for v in value:
#         keys.append(v.flatten()) # Преобразование вектора в одномерный массив
#         labels.append(key)
#
# # Обучение классификатора KNN
# classifier = KNeighborsClassifier(n_neighbors=2)
# labels = np.array(labels)
# classifier.fit(keys, labels.reshape(-1))
#
# # Получение метки объекта для найденного ключа
#
# # Получение метки для найденного ключа
# predicted = classifier.predict(face)
#
# # Сопоставление меток и возврат результата
#
# # Соответствующий человек, найденный по метке
# match = database[predicted[0]]
#
# # Вывод результата на экран
# predicted = classifier.predict(face)
# print('Найденный класс:', predicted[0])
