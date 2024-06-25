""" Module for simple auto-encoder. """

import torch
from torch import nn


class Autoencoder(nn.Module):
    """ Simple auto-encoder. """

    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """ Forward pass. """

        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x: torch.Tensor):
        """ Encode data. """

        return self.encoder(x)
