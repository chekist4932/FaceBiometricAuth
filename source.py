import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import cv2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Нормализация пикселя
])


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)


def image_shower(images, labels, n=4):
    plt.figure(figsize=(12, 12))
    for i, image in enumerate(images[:n]):
        image = image.to(torch.float32)  # Приведение к типу float32
        image /= 255.0
        plt.subplot(n, n, i + 1)
        image = image / 2 + 0.5
        plt.imshow(image.numpy().transpose((1, 2, 0)).squeeze())
    print("Real Labels: ", ' '.join('%5s' % classes[label] for label in labels[:n]))


num_classes_own = 31
classes = [f"Student{number}" for number in range(num_classes_own)]
classes.append('Alien')
num_classes = num_classes_own + 1

# dir = 'dataset/train/0/IMG_4970.jpg'
#
# image = cv2.imread(dir, cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image.astype(np.float32)
# image = image / 255.0
# image = image.transpose((2, 0, 1))
#
# tensor_image = torch.from_numpy(image)
# # plt.imshow(image)
#
# # plt.pause(3)
# print(image)
#
# im = Image.open(dir)
# im = transform(im)
# print(im)
