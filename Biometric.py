import torch.nn as nn


class BiometricCode(nn.Module):
    def __init__(self):
        super(BiometricCode, self).__init__()

        # Existing architecture for feature extraction
        self.feature_extractor = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # New layers for generating cryptographic key
        self.key_generator = nn.Sequential(
            nn.Linear(in_features=64 * 28 * 28, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=256)
        )

    def forward(self, tensor_image):
        face_feature = self.feature_extractor(tensor_image)
        face_feature = face_feature.view(face_feature.size(0), -1)
        key = self.key_generator(face_feature)
        # key_bytes = key.view(-1, 8).byte()  # Преобразование в байты
        return key
