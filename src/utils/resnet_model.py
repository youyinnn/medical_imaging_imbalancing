import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, output_features, weights=models.ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        model = models.resnet18(
            weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, output_features, weights=models.ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()
        model = models.resnet50(
            weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


class ResNet34(nn.Module):
    def __init__(self, output_features, weights=models.ResNet34_Weights.IMAGENET1K_V1):
        super().__init__()
        model = models.resnet34(
            weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)
