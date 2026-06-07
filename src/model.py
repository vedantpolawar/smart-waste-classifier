import torch.nn as nn
from torchvision.models import efficientnet_b0

def get_model():
    model = efficientnet_b0(weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 6)
    )

    return model