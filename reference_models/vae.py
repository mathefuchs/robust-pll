""" Module for VAE. """

import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    """ Variational auto-encoder implementation. """

    def __init__(
        self, num_features: int = 28 * 28,
        hidden_layer: int = 512, bottleneck: int = 48,
    ):
        super().__init__()
        self.reconstruction_loss = nn.BCELoss(reduction="sum")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_layer),
            nn.ReLU(),
        )
        self.mean_encoder = nn.Linear(hidden_layer, bottleneck)
        self.var_encoder = nn.Linear(hidden_layer, bottleneck)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, num_features),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor, compute_loss: bool = True):
        """ Forward pass. """

        # Encoding
        inputs_enc = self.encoder(inputs)
        mu = self.mean_encoder(inputs_enc)
        log_var = self.var_encoder(inputs_enc)
        if not compute_loss:
            return None, mu, log_var

        # Sampling
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z_sampled = eps.mul(std).add_(mu)

        # Decoding
        inputs_dec = self.decoder(z_sampled)

        # Loss computation
        recon_loss = self.reconstruction_loss(
            inputs_dec, inputs) / inputs_dec.shape[0]
        kl_loss = torch.mean(
            (mu ** 2 + torch.exp(log_var) - 1 - log_var) / 2)
        return recon_loss + kl_loss, mu, log_var
