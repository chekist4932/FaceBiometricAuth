import torch
import torch.nn as nn
from torchvision import models


class ExtraLayers(nn.Module):
    def __init__(self):
        super(ExtraLayers, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            #
            # nn.Sigmoid()
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
