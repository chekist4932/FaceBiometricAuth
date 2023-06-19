import pprint

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.cuda.amp import autocast

import os
from tqdm.auto import tqdm

from source import transform, image_shower, imshow, classes, num_classes
from datasets import train_loader, test_loader, test_set
from model import device, model


def save_model(model_name='model.pth'):
    dir_to_save = os.path.join('models', model_name)
    torch.save(model.state_dict(), dir_to_save)


def load_model(model_name='model.pth'):
    dir_to_load = os.path.join('models', model_name)
    model.load_state_dict(torch.load(dir_to_load))


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
                        save_model(model_name='best_model')

                loss.backward()

                loss_item = loss.item()
                loss_val += loss_item

                optimizer.step()
                progress.set_description(f"Epoch: {epoch} | Training loss(iter): {str(loss_item)[:6]}")
                progress.update()
            progress.set_description(f"Epoch: {epoch} | Training loss(total): {str(loss_val / len(train_loader))[:6]}")


def is_correct(output, label):
    output = output.to(torch.uint8)
    label = label.to(torch.uint8)
    if torch.mean(torch.isclose(output, label, rtol=0.01, atol=0.01).float()).item() == 1.0:
        return True
    else:
        return False


def test_model():
    correct = 0.0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            for output, label in zip(outputs, labels):
                correct += torch.mean(torch.isclose(output, label, rtol=0.1, atol=0.1).float())

                output = output.to(torch.uint8)
                label = label.to(torch.uint8)

                # _, predicted = torch.argmax(output.data)
                # correct += torch.mean((output == label).float()).item()
                total += 1

    print(f"Accuracy: {100 * correct / total}")


def test_model_frr():
    FRR = 0.0  # False Reject Rate
    total_correct_access = 0.0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            for output, label in zip(outputs, labels):

                correct = is_correct(output, label)
                if correct:
                    total_correct_access += 1
                else:
                    FRR += 1
                total += 1

    print(f"FRR: {100 * FRR / total:.4f} | Accuracy: {100 * total_correct_access / total}")
    return FRR, total_correct_access


def test_model_far():
    total = 0.0
    total_correct_denied = 0.0
    FAR = 0.0
    tens_to_test_far = torch.load('tens_to_test_far.pth')
    keys = torch.load('middle_tensor_dict.pth')
    for class_id, tensors in tens_to_test_far.items():
        for key_id in keys.keys():
            if class_id == key_id:
                continue

            for tens in tensors:
                correct = is_correct(tens, keys[key_id])
                if correct:
                    FAR += 1
                else:
                    total_correct_denied += 1
                total += 1
    print(f"FAR: {100 * FAR / total:.4f} | Accuracy: {100 * total_correct_denied / total}")
    return FAR, total_correct_denied


lr = 0.0001
wd = 0.0001
EPOCHS = 50

loss_fn = nn.MSELoss().to(device)
best_loss = float('inf')

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

use_amp = False
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    summary(model, (3, 224, 224))
    # train_model(best_loss)
    load_model(model_name='model_test_resnet_50.pth')
    # test_model()
    far, tn = test_model_far()
    frr, tp = test_model_frr()
    accuracy = (tn + tp) / (frr + far + tn + tp)
    print(f'Accuracy: {accuracy}')
    # tens = torch.load('middle_tensor_dict.pth')
    # with torch.no_grad():
    #     model.eval()
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(images)
    #         for output, label in zip(outputs, labels):
    #             class_id_out = torch.argmax(output.data).item()
    #             class_id_lab = torch.argmax(output.data).item()
    #             pass

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
