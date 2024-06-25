""" Module for CroSel. """

import itertools
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
from reference_models.lenet import LeNet
from reference_models.mlp import MLP


class CroSel(PllBaseClassifier):
    """
    CroSel by Tian et al. (2024),
    "CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning".
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
    ) -> None:
        self.rng = rng
        self.device = torch.device("cpu")
        self.debug = debug
        self.adv_eps = adv_eps
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(int(self.rng.integers(1000)))
        torch.manual_seed(int(self.rng.integers(1000)))
        self.model1: Optional[MLP] = None
        self.model2: Optional[MLP] = None

    def _warm_up_with_cc(
        self, model: nn.Module, num_inst: int, num_classes: int,
        data_loader: DataLoader,
    ) -> torch.Tensor:
        # Optimizer
        model.train()
        optimizer = torch.optim.Adam(model.parameters())

        # Memory Bank
        memory_bank = torch.zeros(
            (1000, num_inst, num_classes), dtype=torch.float32)

        # Training loop
        class_norm = 1 / (2 ** (num_classes - 1) - 1)
        for t in range(10):
            for idx_batch, x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward-backward pass
                model_out = model(x_batch)
                probs = y_batch * model_out
                loss = torch.mean(-torch.log(
                    class_norm * torch.sum(probs, dim=1) + 1e-10
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store to memory bank
                memory_bank[t, idx_batch, :] = model_out.cpu().clone().detach()

        return memory_bank

    @torch.no_grad
    def _get_selection_mask_from_memory_bank(
        self, memory_bank: torch.Tensor, y_train: torch.Tensor,
        curr_epoch: int,
    ) -> torch.Tensor:
        thresh = 0.9

        beta_1 = torch.count_nonzero(torch.isclose(
            memory_bank[curr_epoch - 1],
            torch.max(memory_bank[curr_epoch - 1],
                      dim=-1, keepdim=True).values,
        ) * y_train, dim=-1) > 0  # N
        beta_2 = (
            torch.argmax(memory_bank[curr_epoch - 1], dim=-1) ==
            torch.argmax(memory_bank[curr_epoch - 2], dim=-1)
        ) * (
            torch.argmax(memory_bank[curr_epoch - 2], dim=-1) ==
            torch.argmax(memory_bank[curr_epoch - 3], dim=-1)
        )  # N
        beta_3 = (
            torch.max(torch.mean(
                memory_bank[curr_epoch - 3:curr_epoch], dim=0,
            ), dim=-1).values > thresh
        )  # N

        # Return mask of selected instances
        return beta_1 * beta_2 * beta_3  # N

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
        num_epoch = 200
        if len(inputs.shape) == 2:
            # Tabular data (Batch, Features)
            self.model1 = MLP(inputs.shape[1], partial_targets.shape[1])
            self.model2 = MLP(inputs.shape[1], partial_targets.shape[1])
            self.device = torch.device("cpu")  # Always fastest on CPU
            batch_size = 256
        elif len(inputs.shape) == 4:
            # Image data (Batch, Channels, Height, Width)
            self.model1 = LeNet(partial_targets.shape[1])
            self.model2 = LeNet(partial_targets.shape[1])
            batch_size = 64
            if torch.cuda.is_available():
                cuda_idx = random.randrange(torch.cuda.device_count())
                self.device = torch.device(f"cuda:{cuda_idx}")
            else:
                self.device = torch.device("cpu")
        else:
            raise ValueError(f"Malformed data of shape {inputs.shape}.")
        self.model1.to(self.device)
        self.model2.to(self.device)
        opt1 = torch.optim.Adam(self.model1.parameters())
        opt2 = torch.optim.Adam(self.model2.parameters())

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, y_train),
            batch_size=batch_size, shuffle=True, generator=self.torch_rng,
        )

        # Warm-up with CC
        warm_up_epochs = 10
        memory_bank1 = self._warm_up_with_cc(
            self.model1, inputs.shape[0], partial_targets.shape[1], data_loader)
        memory_bank2 = self._warm_up_with_cc(
            self.model2, inputs.shape[0], partial_targets.shape[1], data_loader)
        lambda_cr = 2.0

        # Train loop
        criterion = nn.CrossEntropyLoss()
        for epoch in self.loop_wrapper(range(num_epoch)):
            mask1 = self._get_selection_mask_from_memory_bank(
                memory_bank1, y_train, epoch + warm_up_epochs)
            mask2 = self._get_selection_mask_from_memory_bank(
                memory_bank2, y_train, epoch + warm_up_epochs)
            count1 = int(torch.count_nonzero(mask1).item())
            count2 = int(torch.count_nonzero(mask2).item())
            frac_labeled1 = count1 / mask1.shape[0]
            frac_labeled2 = count2 / mask1.shape[0]
            lambda_d1 = (1 - frac_labeled1) * lambda_cr
            lambda_d2 = (1 - frac_labeled2) * lambda_cr
            y_hat1 = torch.argmax(
                memory_bank1[warm_up_epochs + epoch - 1], dim=-1)
            y_hat2 = torch.argmax(
                memory_bank2[warm_up_epochs + epoch - 1], dim=-1)
            drop_last1 = count1 % batch_size in [1, 2, 3, 4, 5]
            drop_last2 = count2 % batch_size in [1, 2, 3, 4, 5]

            if count1 >= 2:
                selected_data_loader1 = DataLoader(
                    TensorDataset(x_train[mask1], y_hat1[mask1]),
                    batch_size=batch_size, shuffle=True, generator=self.torch_rng,
                    drop_last=drop_last1,
                )
            else:
                selected_data_loader1 = [(None, None)]
            if count2 >= 2:
                selected_data_loader2 = DataLoader(
                    TensorDataset(x_train[mask2], y_hat2[mask2]),
                    batch_size=batch_size, shuffle=True, generator=self.torch_rng,
                    drop_last=drop_last2,
                )
            else:
                selected_data_loader2 = [(None, None)]

            # Model 1
            for (
                (idx_batch, x_batch, y_batch), (x_batch_sel, y_hat_sel),
            ) in zip(data_loader, itertools.cycle(selected_data_loader2)):
                # Ground-truth part
                if frac_labeled2 > 0.05 and count2 >= 2:
                    x_batch_sel = x_batch_sel.to(self.device)
                    y_hat_sel = y_hat_sel.to(self.device)
                    probs = self.model1(x_batch_sel)
                    loss_hat = criterion(probs, y_hat_sel)
                else:
                    loss_hat = 0.0

                # PLL part
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                weights_batch = memory_bank2[
                    warm_up_epochs + epoch - 1, idx_batch, :].clone().detach().to(self.device)
                probs_pll = self.model1(x_batch)
                pll_loss = torch.mean(torch.sum(
                    weights_batch * -torch.log(probs_pll + 1e-10), dim=-1,
                ))
                loss = loss_hat + lambda_d1 * pll_loss

                opt1.zero_grad()
                loss.backward()
                opt1.step()

                memory_bank2[
                    warm_up_epochs + epoch, idx_batch, :,
                ] = probs_pll.cpu().clone().detach()

            # Model 2
            for (
                (idx_batch, x_batch, y_batch), (x_batch_sel, y_hat_sel),
            ) in zip(data_loader, itertools.cycle(selected_data_loader1)):
                # Ground-truth part
                if frac_labeled1 > 0.05 and count1 >= 2:
                    x_batch_sel = x_batch_sel.to(self.device)
                    y_hat_sel = y_hat_sel.to(self.device)
                    probs = self.model2(x_batch_sel)
                    loss_hat = criterion(probs, y_hat_sel)
                else:
                    loss_hat = 0.0

                # PLL part
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                weights_batch = memory_bank1[
                    warm_up_epochs + epoch - 1, idx_batch, :].clone().detach().to(self.device)
                probs_pll = self.model2(x_batch)
                pll_loss = torch.mean(torch.sum(
                    weights_batch * -torch.log(probs_pll + 1e-10), dim=-1,
                ))
                loss = loss_hat + lambda_d2 * pll_loss

                opt2.zero_grad()
                loss.backward()
                opt2.step()

                memory_bank1[
                    warm_up_epochs + epoch, idx_batch, :,
                ] = probs_pll.cpu().clone().detach()

        # Return results
        res = 0.5 * (
            memory_bank1[warm_up_epochs + num_epoch - 1]
            + memory_bank2[warm_up_epochs + num_epoch - 1]
        )
        return SplitResult.from_scores(self.rng, res.cpu().numpy())

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.model1 is None and self.model2 is not None:
            raise ValueError()

        inputs = flatten_if_image(inputs)
        if len(inputs.shape) == 2:
            batch_size = 256
        elif len(inputs.shape) == 4:
            batch_size = 64
        else:
            raise ValueError(f"Malformed data of shape {inputs.shape}.")
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model1.eval()
        self.model2.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                x_batch = generate_adversarial(
                    lambda x: 0.5 * (
                        self.model1.logits(x) + self.model2.logits(x)
                    ), x_batch, self.adv_eps,
                )
                out1 = self.model1(x_batch).cpu().numpy()
                out2 = self.model2(x_batch).cpu().numpy()
                all_results.append(0.5 * (out1 + out2))
            test_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, test_probs)
