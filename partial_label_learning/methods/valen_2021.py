"""
    Module for VALEN.
    Code modified from "https://github.com/palm-ml/valen".
    License: Mulan PSL v2
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import lil_array
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from partial_label_learning.adversarial import generate_adversarial
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class MlpFeature(nn.Module):
    """ MLP with feature extraction. """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass. """

        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        feature = self.relu3(out)
        out = self.fc4(feature)
        return feature, out


class Valen(PllBaseClassifier):
    """
    Valen by Xu et al.,
    "Instance-Dependent Partial Label Learning"
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
    ) -> None:
        self.rng = rng
        self.adv_eps = adv_eps
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(int(self.rng.integers(1000)))
        torch.manual_seed(int(self.rng.integers(1000)))
        self.model: Optional[MlpFeature] = None
        if torch.cuda.is_available():
            cuda_idx = random.randrange(torch.cuda.device_count())
            self.device = torch.device(f"cuda:{cuda_idx}")
        else:
            self.device = torch.device("cpu")

    def _warmup(
        self, data_loader: DataLoader, loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Optimizer
        if self.model is None:
            raise ValueError()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Training loop
        self.model.train()
        for _ in range(10):
            for idx, inputs_i, partial_targets_i, w_ij, _ in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward-backward pass
                phi, probs = self.model(inputs_i)
                loss, new_rel = self._partial_loss(probs, w_ij)
                loss_weights[idx] = new_rel.cpu()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract features
        self.model.eval()
        feature_extracted = torch.zeros(
            (loss_weights.shape[0], phi.shape[-1]), dtype=torch.float32)
        with torch.no_grad():
            for idx, inputs_i, _, _, _ in data_loader:
                inputs_i = inputs_i.to(self.device)
                feature_extracted[idx, :] = self.model(inputs_i)[0].cpu()

        return feature_extracted

    def _partial_loss(
        self, output1: torch.Tensor,
        target: torch.Tensor, eps: float = 1e-10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = F.softmax(output1, dim=1)
        ce_loss = target * -torch.log(output + eps)
        loss = torch.sum(ce_loss) / ce_loss.size(0)
        return loss, self._revised_target(output, target)

    def _revised_target(
        self, output: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        revised_y = target.clone().detach()
        revised_y[revised_y > 0] = 1
        revised_y = revised_y * (output.clone().detach() + 1e-10)
        revised_y = revised_y / revised_y.sum(dim=1, keepdim=True)
        return revised_y

    def _alpha_loss(
        self, alpha: torch.Tensor, prior_alpha: torch.Tensor,
    ) -> torch.Tensor:
        kld = (
            torch.mvlgamma(alpha.sum(1), p=1) -
            torch.mvlgamma(alpha, p=1).sum(1) -
            torch.mvlgamma(prior_alpha.sum(1), p=1) +
            torch.mvlgamma(prior_alpha, p=1).sum(1) + (
                (alpha - prior_alpha) * (
                    torch.digamma(alpha) -
                    torch.digamma(
                        alpha.sum(dim=1, keepdim=True).expand_as(alpha))
                )
            ).sum(1)
        )
        return kld.mean()

    def _dot_product_decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(z, z.T))

    def _subselect_lilarray(
        self, sparse: lil_array, idx: torch.Tensor,
    ) -> torch.Tensor:
        idx_arr = idx.cpu().numpy()
        dense_subsample = sparse[
            idx_arr, :][:, idx_arr].todense().copy()  # type: ignore
        return torch.tensor(
            dense_subsample, dtype=torch.float32, device=self.device)

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

        # Problem instance sizes
        inputs = flatten_if_image(inputs)
        num_feats = inputs.shape[1]
        num_classes = partial_targets.shape[1]
        num_epoch = 200

        # Create model
        self.model = MlpFeature(num_feats, num_feats, num_classes)
        encoder = MlpFeature(num_feats, num_feats, num_classes)
        batch_size = 256

        # Decoder and devices
        decoder = nn.Sequential(
            nn.Linear(num_classes, num_feats), nn.ReLU(),
            nn.Linear(num_feats, num_feats),
        )
        self.model.to(self.device)
        encoder.to(self.device)
        decoder.to(self.device)

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        d_array = loss_weights.clone().detach()
        prior_alpha = torch.ones(
            (1, num_classes), dtype=torch.float32, device=self.device)
        data_loader = DataLoader(
            TensorDataset(
                train_indices, x_train, y_train, loss_weights, d_array,
            ),
            batch_size=batch_size, shuffle=True, generator=self.torch_rng,
        )

        # Warm-up
        self.model.train()
        feature_extracted = self._warmup(data_loader, loss_weights)

        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(feature_extracted.numpy())
        nn_indices = knn.kneighbors(return_distance=False)
        adj_matrix = lil_array((inputs.shape[0], inputs.shape[0]), dtype=float)
        for i, nn_row in enumerate(nn_indices):
            adj_matrix[i, i] = 1
            for neigh in nn_row:
                adj_matrix[neigh, i] = 1
        adj_matrix = adj_matrix / adj_matrix.sum(axis=1).reshape(-1, 1)
        adj_matrix = adj_matrix.tolil()
        assert isinstance(adj_matrix, lil_array)

        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(encoder.parameters()) +
            list(decoder.parameters())
        )

        # Training loop
        self.model.train()
        encoder.train()
        decoder.train()
        for _ in self.loop_wrapper(range(num_epoch)):
            for idx, inputs_i, partial_targets_i, w_ij, d_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)
                d_ij = d_ij.to(self.device)

                # Forward pass
                _, outputs = self.model(inputs_i)
                _, alpha = encoder(inputs_i)

                # Model and encoder loss
                loss_d, new_d = self._partial_loss(alpha, w_ij)
                loss_obj, new_o = self._partial_loss(outputs, d_ij)

                # Alpha loss
                s_alpha = F.softmax(alpha, dim=1)
                revised_alpha = torch.zeros_like(w_ij, device=self.device)
                revised_alpha[w_ij > 0] = 1.0
                s_alpha = s_alpha * revised_alpha
                s_alpha_sum = s_alpha.clone().detach().sum(dim=1, keepdim=True)
                s_alpha = s_alpha / s_alpha_sum + 1e-2
                alpha = torch.exp(alpha / 4)
                alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
                loss_alpha = self._alpha_loss(alpha, prior_alpha)

                # Reconstruction loss
                dirichlet_sample_machine = \
                    torch.distributions.dirichlet.Dirichlet(s_alpha)
                d: torch.Tensor = dirichlet_sample_machine.rsample()  # type: ignore
                x_hat: torch.Tensor = decoder(d)
                x_hat = x_hat.view(inputs_i.shape)
                a_hat = F.softmax(self._dot_product_decode(d), dim=1)
                loss_rec_x = F.mse_loss(x_hat, inputs_i)
                loss_rec_y = 0.01 * F.binary_cross_entropy_with_logits(
                    d, partial_targets_i)
                loss_rec_a = F.mse_loss(
                    a_hat, self._subselect_lilarray(adj_matrix, idx))
                loss_rec = loss_rec_x + loss_rec_y + loss_rec_a

                # Backward pass on final loss
                final_loss = loss_d + loss_obj + loss_alpha + loss_rec
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

                # Update weights
                new_d = self._revised_target(d, new_d)
                d_array[idx, :] = new_d.clone().detach().cpu()
                loss_weights[idx, :] = new_o.clone().detach().cpu()

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
                    lambda x: self.model(x)[1], x_batch, self.adv_eps)  # type: ignore
                all_results.append(
                    F.softmax(self.model(x_batch)[1], dim=1).cpu().numpy())
            test_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, test_probs)
