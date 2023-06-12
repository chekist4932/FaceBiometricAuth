import torch
import torch.nn as nn
import torchvision.models as models


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
        self.extract_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(1, 1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128, 2),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256, 2),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512, 2),
            nn.AvgPool2d(2)
        )
        self.classification = nn.Linear(32 * (64 // 4) * (64 // 4), 46)
        # Fully connected layers for generating the code
        self.key_generation = nn.Sequential(
            nn.Linear(in_features=32 * (64 // 4) * (64 // 4), out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256)
        )

    def forward(self, tensor_image):
        features = self.extract_feature(tensor_image)
        features = features.view(features.size(0), -1)
        x = self.classification(features)
        # key = self.key_generation(features)
        # return key
        return x