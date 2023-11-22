import torch.nn as nn
from torchvision import models


class ISICResNet18(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)
