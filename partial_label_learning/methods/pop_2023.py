""" Module for POP. """

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.adversarial import generate_adversarial
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult
from reference_models.mlp import MLP


class Pop(PllBaseClassifier):
    """
    POP by Xu et al.,
    "Progressive Purification for Instance-Dependent Partial Label Learning"
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
        self.roll_window = 5
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
        self.model = MLP(inputs.shape[1], partial_targets.shape[1])
        self.model.to(self.device)

        # Data preparation
        num_samples = inputs.shape[0]
        num_classes = partial_targets.shape[1]
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

        # Rolling window confidences
        curr_correction_label_matrix = y_train.clone().detach()
        label_confidences_window = torch.zeros(
            (self.roll_window, num_samples, num_classes),
            dtype=torch.float32,
        )
        theta = 0.001
        inc = 0.001

        # Training loop
        for epoch in self.loop_wrapper(range(200)):
            # Mini-batch training
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

            # Purify labels
            label_confidences_window[epoch % self.roll_window] = loss_weights
            if epoch >= 20 and epoch % self.roll_window == 0:
                # Save previous state
                prev_correction_label_matrix = curr_correction_label_matrix.clone().detach()

                # Label purification
                label_confidences = torch.mean(label_confidences_window, dim=0)
                purify_mask = label_confidences / torch.max(
                    label_confidences, dim=1, keepdim=True)[0] < theta
                nonzero_mask = curr_correction_label_matrix != 0
                still_enough_pos_entries = torch.count_nonzero(
                    nonzero_mask & ~purify_mask, dim=1).view(-1, 1) > 0
                curr_correction_label_matrix[
                    purify_mask & still_enough_pos_entries] = 0

                # Update label weights
                new_label_matrix = label_confidences * curr_correction_label_matrix
                loss_weights[:] = new_label_matrix / torch.sum(
                    new_label_matrix, dim=1, keepdim=True)

                # Increase threshold if too few changes
                if theta < 0.4 and torch.sum(torch.not_equal(
                    prev_correction_label_matrix,
                    curr_correction_label_matrix,
                )) < 0.0001 * num_samples * num_classes:
                    theta *= (inc + 1)

        # Return results
        return SplitResult.from_scores(self.rng, loss_weights.numpy())

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
                all_results.append(
                    self.model(x_batch).to("cpu").numpy())
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
