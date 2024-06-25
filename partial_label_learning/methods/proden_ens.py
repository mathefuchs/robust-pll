""" Module for PRODEN ensemble. """

import random
from typing import List, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.adversarial import generate_adversarial
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult
from reference_models.mlp import MLP


def _train_model(
    inputs: np.ndarray, partial_targets: np.ndarray,
    model: MLP, device: torch.device, seed: int,
):
    """ Trains a single model. """

    # Data preparation
    x_train = torch.tensor(inputs, dtype=torch.float32)
    y_train = torch.tensor(partial_targets, dtype=torch.float32)
    train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
    loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
    loss_weights /= loss_weights.sum(dim=1, keepdim=True)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)
    torch.manual_seed(seed)
    data_loader = DataLoader(
        TensorDataset(train_indices, x_train, y_train, loss_weights),
        batch_size=256, shuffle=True, generator=torch_rng,
    )

    # Optimizer
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for _ in range(200):
        for idx, inputs_i, partial_targets_i, w_ij in data_loader:
            # Move to device
            inputs_i = inputs_i.to(device)
            partial_targets_i = partial_targets_i.to(device)
            w_ij = w_ij.to(device)

            # Forward-backward pass
            probs = model(inputs_i)
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

    return loss_weights.detach().cpu().numpy(), model


class ProdenEnsemble(PllBaseClassifier):
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
        self.models: List[MLP] = []
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
        self.models = [
            MLP(inputs.shape[1], partial_targets.shape[1]).to(self.device)
            for _ in range(5)
        ]
        res = Parallel(n_jobs=5)(
            delayed(_train_model)(
                inputs, partial_targets, model,
                self.device, self.rng.choice(1000),
            )
            for model in self.models
        )
        model_weights: List[Tuple[np.ndarray, MLP]] = res  # type: ignore
        self.models = [
            m for _, m in model_weights
        ]
        weight_res: np.ndarray = model_weights[0][0]
        for w, _ in model_weights[1:]:
            weight_res += w
        weight_res *= 0.2

        # Return results
        return SplitResult.from_scores(self.rng, weight_res)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        inputs = flatten_if_image(inputs)
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=256, shuffle=False,
        )

        # Switch to eval mode
        for model in self.models:
            model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                x_batch_adv = generate_adversarial(
                    self.models[0].logits, x_batch, self.adv_eps)
                res = self.models[0](x_batch_adv).cpu().numpy()
                for i in range(1, 5):
                    x_batch_adv = generate_adversarial(
                        self.models[i].logits, x_batch, self.adv_eps)
                    res += self.models[i](x_batch_adv).cpu().numpy()
                res *= 0.2
                all_results.append(res)
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
