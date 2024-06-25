""" Module for simple LeNet architecture. """

import torch
from torch import nn


class LeNet(nn.Module):
    """ LeNet architecture. """

    def __init__(self, num_classes: int, last_layer: str = "softmax"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 1 input, 6 output channels
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(6, 16, kernel_size=5),  # 6 input, 16 output channels
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
            self._last_layer(last_layer),
        )

    def _last_layer(self, last_layer: str) -> nn.Module:
        if last_layer == "softmax":
            return nn.Softmax(dim=1)
        if last_layer == "relu":
            return nn.ReLU()
        if last_layer == "none":
            return nn.Identity()
        raise ValueError()

    def forward(self, inputs: torch.Tensor):
        """ Forward pass. """

        inputs = self.features(inputs)
        inputs = inputs.view(-1, 16 * 4 * 4)
        inputs = self.classifier(inputs)
        return inputs
