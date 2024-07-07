from torch import nn, optim
from torchvision import models
import torch
import torchmetrics
class resModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", average="macro", num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", average="macro", num_classes=num_classes)
        self.f1score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x