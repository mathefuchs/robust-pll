""" Module for Evidential Deep Learning using Fully-Supervised Data. """

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.adversarial import generate_adversarial
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.methods.proden_2020 import Proden
from partial_label_learning.result import SplitResult
from reference_models.mlp import MLP


class ProdenEdl(PllBaseClassifier):
    """
    Evidential Deep Learning using Fully-Supervised Data.
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.adv_eps = adv_eps
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        torch.manual_seed(int(self.rng.integers(1000)))
        self.model: Optional[MLP] = None
        self.num_classes: Optional[int] = None
        if torch.cuda.is_available():
            cuda_idx = random.randrange(torch.cuda.device_count())
            self.device = torch.device(f"cuda:{cuda_idx}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def _kl_divergence_pq(
        self, alpha: torch.Tensor, beta: torch.Tensor,
    ) -> torch.Tensor:
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        sum_beta = torch.sum(beta, dim=1, keepdim=True)
        dg_alpha = torch.special.digamma(alpha)  # pylint: disable=not-callable
        dg_sum = torch.special.digamma(  # pylint: disable=not-callable
            sum_alpha)

        gamma_sum = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
        sum_gamma = torch.sum(
            torch.lgamma(beta) - torch.lgamma(alpha), dim=1, keepdim=True,
        )
        dg = torch.sum(
            (alpha - beta) * (dg_alpha - dg_sum), dim=1, keepdim=True,
        )
        return gamma_sum + sum_gamma + dg

    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        return self._kl_divergence_pq(alpha, torch.ones_like(alpha))

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

        # Use PRODEN for disambiguation
        inputs = flatten_if_image(inputs)
        proden = Proden(self.rng, self.debug, self.adv_eps)
        y_true = proden.fit(inputs, partial_targets).pred
        y_targets = np.zeros_like(partial_targets)
        for i, j in enumerate(y_true):
            y_targets[i, j] = 1

        # Model
        self.num_classes = partial_targets.shape[1]
        self.model = MLP(inputs.shape[1], self.num_classes, "relu")
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters())

        # Data preparation
        torch_rng = torch.Generator()
        torch_rng.manual_seed(int(self.rng.integers(1000)))
        data_loader = DataLoader(
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(y_targets, dtype=torch.float32),
            ),
            batch_size=256, shuffle=True, generator=torch_rng,
        )

        # Training loop
        self.model.train()
        for epoch in self.loop_wrapper(range(200)):
            annealing_coef = np.clip(epoch / 100, 0, 1)

            # Infer labeling from relevancy matrix
            for x_batch, y_batch in data_loader:
                # Move to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                evidence = self.model(x_batch)
                alpha = evidence + 1
                sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
                prob_pred = alpha / sum_alpha

                # Compute loss
                loss_err = torch.sum(
                    (y_batch - prob_pred) ** 2,
                    dim=1, keepdim=True,
                )
                loss_var = torch.sum(
                    prob_pred * (1 - prob_pred) / (sum_alpha + 1),
                    dim=1, keepdim=True,
                )
                loss_kl_div = annealing_coef * self._kl_divergence(
                    (1 - y_batch) * evidence + 1)
                loss = torch.mean(loss_err + loss_var + loss_kl_div)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Inference
        inference_loader = DataLoader(
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(y_targets, dtype=torch.float32),
            ),
            batch_size=256, shuffle=False,
        )
        self.model.eval()
        prob_results = []
        unc_results = []
        with torch.no_grad():
            for x_batch, y_batch in inference_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Inference
                evd_test = y_batch * self.model(x_batch)
                alpha_test = evd_test + 1
                s_alpha = torch.sum(alpha_test, dim=1, keepdim=True)

                # Compute probabilities, belief, and uncertainty
                prob_results.append((alpha_test / s_alpha).cpu().numpy())
                unc_results.append(torch.flatten(
                    self.num_classes / s_alpha).cpu().numpy())

        return SplitResult.from_scores_with_uncertainty(
            self.rng, np.vstack(prob_results), np.concatenate(unc_results),
        )

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
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
            ),
            batch_size=256, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        prob_results = []
        unc_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                x_batch = generate_adversarial(
                    self.model.logits, x_batch, self.adv_eps)

                # Inference
                evd_test = self.model(x_batch)
                alpha_test = evd_test + 1
                s_alpha = torch.sum(alpha_test, dim=1, keepdim=True)

                # Compute probabilities, belief, and uncertainty
                prob_results.append((alpha_test / s_alpha).cpu().numpy())
                unc_results.append(torch.flatten(
                    self.num_classes / s_alpha).cpu().numpy())

        return SplitResult.from_scores_with_uncertainty(
            self.rng, np.vstack(prob_results), np.concatenate(unc_results),
        )
