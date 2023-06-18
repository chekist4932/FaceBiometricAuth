import pprint

import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
from torch.cuda.amp import autocast

import numpy as np
from PIL import Image

import os
from tqdm.auto import tqdm, trange

from source import transform, image_shower, imshow, classes, num_classes
from datasets import train_loader, test_loader


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
model = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
model.to(device)


extra_layers = ExtraLayers()  # encoder

model.fc = extra_layers.to(device)

lr = 0.0001
wd = 0.0001
EPOCHS = 50
# tens = {f'{i}': [] for i in range(num_classes)}
# middle_tensors = {f'{i}': [] for i in range(num_classes)}
# loss_fn = nn.CrossEntropyLoss().to(device)
loss_fn = nn.MSELoss().to(device)
best_loss = float('inf')

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

use_amp = False
torch.backends.cudnn.benchmark = True


def train_model(best_loss):
    for epoch in range(EPOCHS):
        loss_val = 0.0
        model.train()
        with tqdm(total=test_loader.__len__(), position=0) as progress:
            for tensor_image, label in train_loader:
                tensor_image = tensor_image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                with autocast(use_amp):
                    output = model(tensor_image)

                    loss = loss_fn(output, label)
                    if loss < best_loss:
                        best_loss = loss
                        save_model(model_name='best_model.pth')

                loss.backward()

                loss_item = loss.item()
                loss_val += loss_item

                optimizer.step()
                progress.set_description(f"Epoch: {epoch} | Training loss(iter): {str(loss_item)[:6]}")
                progress.update()
            progress.set_description(f"Epoch: {epoch} | Training loss(total): {str(loss_val / len(train_loader))[:6]}")


def test_model():
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            image, label = data[0].to(device), data[1].to(device)
            output = model(image)
            # _, predicted = torch.max(output.data, 1)
            test_loss += loss_fn(output, label).item()
            correct += torch.mean(torch.isclose(output, label, rtol=0.1, atol=0.1).float())
            print(f'{torch.dist(output[0], label[0])}\n--------------------')
            # total += label.size(0)
            # correct += (output == label).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f} | Accuracy: {100 * correct / len(test_loader)}")


def save_model(model_name='model.pth'):
    dir_to_save = os.path.join('models', model_name)
    torch.save(model.state_dict(), dir_to_save)


def load_model(model_name='model.pth'):
    dir_to_load = os.path.join('models', model_name)
    model.load_state_dict(torch.load(dir_to_load))


def label_check():
    images, labels = next(iter(test_loader))
    outputs = model(images.to(device))

    _, predicted = torch.max(outputs, 1)
    count = 0
    for image, label in zip(images, labels):
        if count < 4:
            imshow(image, label)
        else:
            break
        count += 1
    print("Real: ", " ".join("%5s" % classes[predict] for predict in labels[:20]))
    print("Predicted: ", " ".join("%5s" % classes[predict] for predict in predicted[:20]))


def euclidean_distance_equal_stat(middle_tensors: dict, tens: dict):
    for key_middle in middle_tensors.keys():
        sum_dist = 0
        dist = []
        for ind_ in range(len(tens[key_middle])):
            dist_val = torch.dist(middle_tensors[key_middle], tens[key_middle][ind_])
            sum_dist += dist_val
            dist.append(dist_val)
        middle_dist = sum_dist / len(dist)
        print(f'Class: {key_middle} | middle: {middle_dist} | max: {max(dist)} | min: {min(dist)}')


def euclidean_distance_diff_stat(middle_tensors: dict, tens: dict):
    sum_dist = 0
    dist = []
    for key_middle in middle_tensors.keys():
        for key_tens in tens.keys():
            if key_middle == key_tens:
                continue
            for i in range(len(tens[key_tens])):
                dist_val = torch.dist(middle_tensors[key_middle], tens[key_tens][i])
                sum_dist += dist_val
                dist.append(dist_val)
    middle_dist = sum_dist / len(dist)
    print(f'diff | middle: {middle_dist} | max: {max(dist)} | min: {min(dist)}')


if __name__ == "__main__":
    # summary(model, (3, 224, 224))
    # train_model(best_loss)
    load_model(model_name='best_model.pth')
    # test_model()
    # save_model(model_name='model_test_resnet_100.pth')
    # load_model(model_name='model.pth')
    # tens = torch.load('tensor_dict.pth')
    #
    # for key in tens.keys():
    #     temp = 0
    #     for tensor in tens[key]:
    #         temp += tensor
    #     middle = temp / len(tens[key])
    #     middle_tensors[key] = middle
    # pprint.pprint(middle_tensors)
    # middle = torch.load('middle_tensor_dict.pth')
    # tens_array = torch.load('tensor_dict.pth')
    # euclidean_distance_equal_stat(middle, tens_array)
    # euclidean_distance_diff_stat(middle, tens_array)

    # load_model()
    # test_model()
    # with torch.no_grad():
    #     model.eval()
    #     # for images, labels in tqdm(train_loader, position=0, total=train_loader.__len__()):
    #     for images, labels in train_loader:
    #         for im, lab in zip(images.to(device), labels.to(device)):
    #             im = im.unsqueeze(0)
    #             out = model(im)
    #             tens[f'{lab}'].append(out)
    #     print(tens)
    #     input("Save?")
    #     torch.save(tens, 'tensor_dict.pth')

# for image_name in os.listdir('dataset/test/0'):
#     image = Image.open(f'dataset/test/0/{image_name}')
#     tensor_image = transform(image)
#     tensor_image.to(device)
#     input_batch = tensor_image.unsqueeze(0)
#     input_batch = input_batch.to(device)
#     with torch.no_grad():
#         model.eval()
#         out = model(input_batch)
#         _, predicted = torch.max(out.data, 1)
#         # input(predicted.sum())
#         print(predicted)
#         print(out)
#         print("----------------------------")
# summary(model, (3, 224, 224))
# train_model()
# test_model()
# save_model()
# load_model(model_name='model_v2.pth')
# summary(model, (3, 224, 224))
# label_check()

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
