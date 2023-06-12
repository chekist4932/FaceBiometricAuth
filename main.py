import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision import transforms
from torchsummary import summary

import os
import numpy as np
from PIL import Image

from source import transform, imshow, image_show
from feature_extraction import ResNet34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = 'dataset'
train = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
trainset = DataLoader(train, batch_size=15, shuffle=True)

test = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
testset = DataLoader(train, batch_size=15, shuffle=True)

# res = []
# for images, labels in testset:
#     for i in labels:
#         res.append(int(i))
# print(set(res))

# for im, tt in zip(images, labels):
#     imshow(im, tt)
# # image_show(images, labels)

lr = 0.001

# model = models.resnet34().to(device)
#
# num_classes = 46  # Новое количество классов
# model.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, num_classes).to(device)
#
# summary(model, (3, 224, 224))
#
# optimizer = optim.Adam(model.parameters(), lr=lr)
# EPOCHS = 10

# faces = os.listdir(dataset_dir)
#
# iter_count = 0
# try:
#     for face in faces:
#         face_dir = dataset_dir + '/' + face
#         image = Image.open(face_dir)
#         tensor_image = transform(image)
#         tensor_image = transform(image)
#
#         input_batch = tensor_image.unsqueeze(0)
#         input_batch = input_batch.to(device)
#
#         feature = model.forward(input_batch)
#
#         print(f'feature:\n{feature}')
#         print(f'feature len: {feature.shape}')
#         iter_count += 1
# except KeyboardInterrupt:
#     print(f'Total iterations: {iter_count}')
#     exit()

# input('Next iteration - ')

# train = datasets.MNIST("", train=True, download=True,
#                        transform=transforms.Compose([transforms.ToTensor()]))
# test = datasets.MNIST("", train=False, download=True,
#                       transform=transforms.Compose([transforms.ToTensor()]))
#
# trainset = torch.utils.data.DataLoader(train, batch_size=15, shuffle=True)
# testset = torch.utils.data.DataLoader(test, batch_size=15, shuffle=True)
#
#
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 86)
#         self.fc2 = nn.Linear(86, 86)
#         self.fc3 = nn.Linear(86, 86)
#         self.fc4 = nn.Linear(86, 10)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return F.log_softmax(x, dim=1)
#
#
# model = NeuralNetwork()
#
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# EPOCHS = 3
# for epoch in range(EPOCHS):
#     for data in trainset:
#         X, y = data
#         model.zero_grad()
#         output = model(X.view(-1, 28 * 28))
#         loss = F.nll_loss(output, y)
#         loss.backward()
#         optimizer.step()
#     print(loss)
#
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testset:
#         data_input, target = data
#         output = model(data_input.view(-1, 784))
#         for idx, i in enumerate(output):
#             if torch.argmax(i) == target[idx]:
#                 correct += 1
#             total += 1
#
# print('Accuracy: %d %%' % (100 * correct / total))
#
# plt.imshow(X[1].view(28, 28))
# plt.show()
#
# print(torch.argmax(model(X[1].view(-1, 784))[0]))
