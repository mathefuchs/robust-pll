""" Multi-Layer Perceptron. """

import torch
from torch import nn


class MLP(nn.Module):
    """ Standard MLP classifier. """

    def __init__(
        self, m_features: int, l_classes: int, last_layer: str = "softmax",
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(m_features, 300), nn.ReLU(),  # Layer 1-2
            nn.BatchNorm1d(300), nn.Linear(300, 300), nn.ReLU(),  # Layer 2-3
            nn.BatchNorm1d(300), nn.Linear(300, 300), nn.ReLU(),  # Layer 3-4
            nn.BatchNorm1d(300), nn.Linear(300, l_classes),  # Layer 4-5
        )
        self.last_layer = self._get_last_layer(last_layer)

    def _get_last_layer(self, last_layer: str) -> nn.Module:
        if last_layer == "softmax":
            return nn.Softmax(dim=1)
        if last_layer == "relu":
            return nn.ReLU()
        if last_layer == "none":
            return nn.Identity()
        raise ValueError()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """ Return the logits. """

        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """

        return self.last_layer(self.mlp(x))
