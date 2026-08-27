import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


NUM_CLASSES = 2


def create_resnet18(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    input_features = model.fc.in_features

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    return model