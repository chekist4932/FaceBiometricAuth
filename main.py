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

import numpy as np
from PIL import Image

import os
from tqdm.autonotebook import tqdm

from source import transform, image_shower, imshow, classes, num_classes
from datasets import trainset, testset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
model.to(device)
model.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, num_classes).to(device)


lr = 0.0001
wd = 0.0001
EPOCHS = 10

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def train_model():
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for iter_num, data in (pbar := tqdm(enumerate(trainset))):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()

            running_loss += loss_item
        pbar.set_description(f'epoch: {epoch}\tloss: {loss_item:.5f}')
        print(f'Epoch: {epoch} | Training loss: {running_loss / len(trainset)}')


def test_model():
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testset:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}")


def save_model(model_name='model.pth'):
    dir_to_save = os.path.join('models', model_name)
    torch.save(model.state_dict(), dir_to_save)


def load_model(model_name='model.pth'):
    dir_to_load = os.path.join('models', model_name)
    model.load_state_dict(torch.load(dir_to_load))


if __name__ == "__main__":
    load_model(model_name='model_v2.pth')
    summary(model, (3, 224, 224))
    # train_model()
    test_model()
    # save_model(model_name='model_v2.pth')
    # load_model(model_name='model_v2.pth')
    # summary(model, (3, 224, 224))
    # for i in range(10):
    #
    #

# summary(model, (3, 224, 224))

# images, labels = next(iter(testset))
# outputs = model(images.to(device))
#
# _, predicted = torch.max(outputs, 1)
# count = 0
# for image, label in zip(images, labels):
#     if count < 4:
#         imshow(image, label)
#     else:
#         break
#     count += 1
# print("Real: ", " ".join("%5s" % classes_[predict] for predict in labels[:20]))
# print("Predicted: ", " ".join("%5s" % classes_[predict] for predict in predicted[:20]))


#

#


# for im, tt in zip(images, labels):
#     imshow(im, tt)
# # image_show(images, labels)


# model = models.resnet34().to(device)
#
# num_classes = 46  # Новое количество классов
# model.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, num_classes).to(device)
#
# summary(model, (3, 224, 224))
#


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
