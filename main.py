import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision import transforms

import os
from PIL import Image

from feature_extraction import ResNet34
from Biometric import BiometricCode

print(f'CUDA: {torch.cuda.is_available()}')
os.environ['TORCH_HOME'] = 'models\\resnet'
# Загрузка изображения
image = Image.open("face.jpg")

# Определение преобразований
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor()  # Преобразование в тензор
])

# Применение преобразований к изображению
tensor_image = preprocess(image)

# Добавление дополнительной размерности для пакета изображений (если необходимо)
tensor_image = tensor_image.unsqueeze(0)
# Вывод полученного тензора
print(tensor_image)
# model = ResNet34Features()
# model = BiometricCode()
model = ResNet34()

feature = model.forward(tensor_image)

print(f'feature:\n{feature}')
print(f'feature len: {feature.shape}')


# train = datasets.MNIST("", train=True, download=True,
#                        transform=transforms.Compose([transforms.ToTensor()]))
# test = datasets.MNIST("", train=False, download=True,
#                       transform=transforms.Compose([transforms.ToTensor()]))
#
# trainset = torch.utils.data.DataLoader(train, batch_size=15, shuffle=True)
# testset = torch.utils.data.DataLoader(test, batch_size=15, shuffle=True)
#
#
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 86)
        self.fc2 = nn.Linear(86, 86)
        self.fc3 = nn.Linear(86, 86)
        self.fc4 = nn.Linear(86, 10)

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
