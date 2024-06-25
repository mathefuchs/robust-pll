""" Module for Cavl. """

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.adversarial import generate_adversarial
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult
from reference_models.mlp import MLP


class Cavl(PllBaseClassifier):
    """
    Cavl by Zhang et al.,
    "Exploiting Class Activation Value for Partial-Label Learning"
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
        self.model: Optional[MLP] = None
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
        self.model = MLP(
            inputs.shape[1], partial_targets.shape[1], last_layer="none")
        self.model.to(self.device)

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        data_loader = DataLoader(
            TensorDataset(x_train, y_train, loss_weights),
            batch_size=256, shuffle=True, generator=self.torch_rng,
        )

        # Optimizer
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Warm-up epochs
        for _ in range(1):
            for inputs_i, _, w_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward-backward pass
                probs = F.softmax(self.model(inputs_i), dim=1)
                loss = torch.mean(torch.sum(
                    w_ij * -torch.log(probs + 1e-10), dim=1,
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Training loop
        for _ in self.loop_wrapper(range(200)):
            for inputs_i, partial_targets_i, _ in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)

                # Obtain CAV
                class_activations = self.model(inputs_i)
                probs = F.softmax(class_activations, dim=1)
                v_j = torch.abs(class_activations - 1) * class_activations
                v_j_restricted_on_s = torch.where(
                    partial_targets_i == 1, v_j, -torch.inf)
                y_j = torch.where(torch.isclose(v_j, torch.max(
                    v_j_restricted_on_s, dim=1, keepdim=True).values), 1.0, 0.0)
                y_j /= torch.sum(y_j, dim=1, keepdim=True)

                # Loss + backward pass
                loss = torch.mean(torch.sum(
                    y_j * -torch.log(probs + 1e-10), dim=1,
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Inference
        self.model.eval()
        inference_loader = DataLoader(
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(partial_targets, dtype=torch.float32),
            ),
            batch_size=256, shuffle=False,
        )
        with torch.no_grad():
            all_results = []
            for x_batch, y_batch in inference_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                probs = F.softmax(self.model(x_batch), dim=1) * y_batch + 1e-10
                all_results.append(probs.cpu().numpy())
            train_probs = np.vstack(all_results)

        # Return results
        return SplitResult.from_scores(self.rng, train_probs)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.model is None:
            raise ValueError()
        inputs = flatten_if_image(inputs)
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
                x_batch = generate_adversarial(
                    self.model.logits, x_batch, self.adv_eps)
                all_results.append(F.softmax(
                    self.model(x_batch), dim=1
                ).to("cpu").numpy())
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
