import torch
import torch.nn as nn
import torchvision.models as models


# class ResNet34Features(nn.Module):
#     def __init__(self):
#         super(ResNet34Features, self).__init__()
#         self.resnet = models.resnet34(pretrained=True)
#         self.resnet.fc = nn.Identity()  # Удаляем последний полносвязный слой
#
#     def forward(self, tensor_image):
#         features = self.resnet(tensor_image)
#         return features


# class ResNet34(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet34, self).__init__()
#         self.resnet = models.resnet34(pretrained=True)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, num_classes)
#
#     def forward(self, x):
#         return self.resnet(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1, 1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, 2)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128, 2)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256, 2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512, 2)
        )

        self.avgpool = nn.AvgPool2d(2)

        # Fully connected layers for generating the code
        self.key_generation = nn.Sequential(
            nn.Linear(in_features=32 * (64 // 4) * (64 // 4), out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.key_generation(x)
        # x2 = self.fc2(x)
        # x3 = self.fc3(x)
        # return x1, x2, x3
        return x1
