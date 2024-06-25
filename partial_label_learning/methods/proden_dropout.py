""" Module for PRODEN with Dropout. """

import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.adversarial import generate_adversarial
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class MlpDropout(nn.Module):
    """ Standard MLP classifier. """

    def __init__(
        self, m_features: int, l_classes: int, last_layer: str = "softmax",
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.05)
        self.mlp = nn.Sequential(
            nn.Linear(m_features, 300), nn.ReLU(),  # Layer 1-2
            nn.BatchNorm1d(300), self.dropout,
            nn.Linear(300, 300), nn.ReLU(),  # Layer 2-3
            nn.BatchNorm1d(300), self.dropout,
            nn.Linear(300, 300), nn.ReLU(),  # Layer 3-4
            nn.BatchNorm1d(300), self.dropout,
            nn.Linear(300, l_classes),  # Layer 4-5
        )
        self.last_layer = self._last_layer(last_layer)

    def _last_layer(self, last_layer: str) -> nn.Module:
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


class ProdenDropout(PllBaseClassifier):
    """
    PRODEN by Lv et al.,
    "Progressive Identification of True Labels for Partial-Label Learning"
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.adv_eps = adv_eps
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(int(self.rng.integers(1000)))
        torch.manual_seed(int(self.rng.integers(1000)))
        self.model: Optional[MlpDropout] = None
        if torch.cuda.is_available():
            cuda_idx = random.randrange(torch.cuda.device_count())
            self.device = torch.device(f"cuda:{cuda_idx}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        inputs = flatten_if_image(inputs)
        self.model = MlpDropout(inputs.shape[1], partial_targets.shape[1])
        self.model.to(self.device)

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, y_train, loss_weights),
            batch_size=256, shuffle=True, generator=self.torch_rng,
        )

        # Optimizer
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Training loop
        for _ in self.loop_wrapper(range(200)):
            for idx, inputs_i, partial_targets_i, w_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward-backward pass
                probs = self.model(inputs_i)
                loss = torch.mean(torch.sum(
                    w_ij * -torch.log(probs + 1e-10), dim=1,
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update weights
                with torch.no_grad():
                    updated_w = partial_targets_i * probs
                    updated_w /= torch.sum(updated_w, dim=1, keepdim=True)
                    loss_weights[idx] = updated_w.to("cpu")

        # Return results
        return SplitResult.from_scores(self.rng, loss_weights.numpy())

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        inputs = flatten_if_image(inputs)
        if self.model is None:
            raise ValueError()
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=256, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                self.model.dropout.eval()
                x_batch = generate_adversarial(
                    self.model.logits, x_batch, self.adv_eps)
                self.model.dropout.train()
                probs = self.model(x_batch).to("cpu").numpy()
                for _ in range(999):
                    probs += self.model(x_batch).to("cpu").numpy()
                probs *= 0.001
                all_results.append(probs)
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
