"""
DenseNet121 and ResNet50 pre-trained on ImageNet, with a new output layer.
"""
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("resnet50", "densenet121")


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name!r}, choose from {MODEL_NAMES}")
    return model


def get_head(model: nn.Module) -> nn.Module:
    """The newly added output layer."""
    return model.fc if hasattr(model, "fc") else model.classifier


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Phase 1: freeze everything except the head. Phase 2: train the whole model."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in get_head(model).parameters():
        p.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
