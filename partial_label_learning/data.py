""" Module for loading data. """

import os
import tarfile
from typing import List

import numpy as np
import torch
import torchvision
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

from partial_label_learning.config import REAL_WORLD_LABEL_TO_PATH
from partial_label_learning.data_notmnist import NotMNIST
from reference_models.mlp import MLP


class Dataset:
    """ A dataset. """

    def __init__(
        self, x_full: np.ndarray, y_full: np.ndarray, y_true: np.ndarray,
    ) -> None:
        self.x_full = x_full
        self.y_full = y_full
        self.y_true = y_true

    def copy(self) -> "Dataset":
        """ Copies the dataset.

        Returns:
            Dataset: The copy.
        """

        return Dataset(
            self.x_full.copy(), self.y_full.copy(), self.y_true.copy(),
        )

    def remove_class(self, holdout_class: int) -> np.ndarray:
        """ Removes a class from the dataset.

        Args:
            holdout_class (int): The class to remove.

        Returns:
            np.ndarray: The holdout samples removed.
        """

        # Extract hold-out samples
        is_holdout = (self.y_true == holdout_class).copy()
        x_holdout = self.x_full[is_holdout].copy()

        # Remove class and associated samples from train set
        self.x_full = self.x_full[~is_holdout].copy()
        self.y_full = self.y_full[~is_holdout][
            :, np.arange(self.y_full.shape[1]) != holdout_class
        ].copy()
        self.y_true = np.where(
            self.y_true > holdout_class,
            self.y_true - 1,
            self.y_true,
        )[~is_holdout].copy()

        return x_holdout

    def create_data_split(
        self, rng: np.random.Generator, train_frac: float = 0.8,
    ) -> "Datasplit":
        """ Creates a random data split. """

        train_ind = rng.choice(
            self.x_full.shape[0], size=int(train_frac * self.x_full.shape[0]),
            replace=False, shuffle=False,
        )
        test_ind = np.setdiff1d(np.arange(self.x_full.shape[0]), train_ind)
        rng.shuffle(train_ind)
        rng.shuffle(test_ind)
        return Datasplit(
            x_train=self.x_full[train_ind].copy(),
            x_test=self.x_full[test_ind].copy(),
            y_train=self.y_full[train_ind].copy(),
            y_true_train=self.y_true[train_ind].copy(),
            y_true_test=self.y_true[test_ind].copy(),
        )


class Datasplit:
    """ A data split. """

    def __init__(
        self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
        y_true_train: np.ndarray, y_true_test: np.ndarray,
    ) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test

        # Normalize
        self.x_train_min = np.min(self.x_train, axis=0)
        self.x_train_max = np.max(self.x_train, axis=0)
        self.x_train_min = np.where(
            self.x_train_min == self.x_train_max, 0, self.x_train_min)
        self.x_train_max = np.where(
            self.x_train_min == self.x_train_max, 1, self.x_train_max)
        self.x_train = self._transform(self.x_train)
        self.x_test = self._transform(self.x_test)

    def _transform(self, x_data: np.ndarray) -> np.ndarray:
        """ Normalizes the given data.

        Args:
            x_data (np.ndarray): The data.

        Returns:
            np.ndarray: The normalized data.
        """

        return (
            (x_data - self.x_train_min) /
            (self.x_train_max - self.x_train_min)
        )

    def copy(self) -> "Datasplit":
        """ Copies the datasplit.

        Returns:
            Datasplit: The copy.
        """

        return Datasplit(
            self.x_train.copy(), self.x_test.copy(), self.y_train.copy(),
            self.y_true_train.copy(), self.y_true_test.copy(),
        )

    def augment_targets(
        self,
        rng: np.random.Generator,
        r_candidates: int,
        percent_partially_labeled: float,
        eps_cooccurrence: float,
    ) -> "Datasplit":
        """ Augments a supervised dataset with random label candidates. """

        # Create co-occurrence pairs
        l_classes = self.y_train.shape[1]
        class_perm = list(map(int, rng.permutation(l_classes)))
        class_perm.append(-1)  # Sentinel if odd number of elements
        class_pairs = list(zip(class_perm[::2], class_perm[1::2]))
        co_occ_classes = {}
        for elem1, elem2 in class_pairs:
            co_occ_classes[elem1] = elem2
            co_occ_classes[elem2] = elem1

        # Determine probabilities of item selection
        eps_cooccurrence = max(eps_cooccurrence, 1 / (l_classes - 1))
        other_prob = (1 - eps_cooccurrence) / (l_classes - 2)

        # Iterate train set and add false-positve labels
        y_train_copy = self.y_train.copy()
        for i in range(y_train_copy.shape[0]):
            # Partially label a percentage of all instances
            if rng.random() < percent_partially_labeled:
                # Compute probabilities for each label
                true_label = int(self.y_true_train[i])
                co_occ_class = co_occ_classes[true_label]
                if co_occ_class != -1:
                    probs = other_prob * np.ones(l_classes)
                    probs[true_label] = 0
                    probs[co_occ_class] = eps_cooccurrence
                else:
                    probs = (1 / (l_classes - 1)) * np.ones(l_classes)
                    probs[true_label] = 0

                # Check that probabilities sum to one
                if np.abs(np.sum(probs) - 1) > 1e-10:
                    raise ValueError("Probabilities must sum to one.")

                # Draw candidates
                candidates = list(map(int, rng.choice(
                    l_classes, replace=False, p=probs, size=r_candidates,
                )))
                y_train_copy[i, candidates] = 1

        return Datasplit(
            self.x_train, self.x_test, y_train_copy,
            self.y_true_train, self.y_true_test,
        )

    def augment_targets_instance_dependent(
        self, rng: np.random.Generator,
    ) -> "Datasplit":
        """ Augments a supervised dataset with instance-dependent noise. """

        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Create model
        torch_rng = torch.Generator()
        torch.manual_seed(int(rng.integers(int(1e6))))
        torch_rng.manual_seed(int(rng.integers(int(1e6))))
        model = MLP(
            self.x_train.reshape(self.x_train.shape[0], -1).shape[1],
            self.y_train.shape[1],
        )
        model.to(device)
        optim = torch.optim.SGD(model.parameters())
        model.train()

        # Prepare data
        x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        data_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=256, shuffle=True, generator=torch_rng,
        )

        # Training loop
        for _ in range(100):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
                y_batch = y_batch.to(device)
                probs = model(x_batch)
                loss = torch.mean(torch.sum(
                    (y_batch - probs) ** 2, dim=1))
                optim.zero_grad()
                loss.backward()
                optim.step()

        # Inference
        model.eval()
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(self.x_train, dtype=torch.float32)),
            batch_size=256, shuffle=False,
        )
        with torch.no_grad():
            all_results = []
            for x_batch in inference_loader:
                x_batch = x_batch[0].view(x_batch[0].shape[0], -1).to(device)
                all_results.append(model(x_batch).cpu().numpy())
            train_probs = np.vstack(all_results)

        # Determine augmentation probabilities
        train_false_probs = (1 - self.y_train) * train_probs
        max_false_probs = np.max(train_false_probs, axis=1, keepdims=True)
        train_false_probs /= np.where(
            max_false_probs > 1e-10, max_false_probs, 1.0)
        mean_false_probs = np.mean(
            train_false_probs, axis=1, keepdims=True)
        train_false_probs = 1.0 * train_false_probs / np.where(
            mean_false_probs > 1e-10, mean_false_probs, 1.0)
        train_false_probs = np.clip(train_false_probs, 0.0, 1.0)

        # Augmentation
        sampler = torch.distributions.binomial.Binomial(
            total_count=1, probs=torch.tensor(train_false_probs))
        sample = sampler.sample()
        y_train_copy = self.y_train.copy()
        y_train_copy[sample == 1] = 1

        return Datasplit(
            self.x_train, self.x_test, y_train_copy,
            self.y_true_train, self.y_true_test,
        )


class DatasplitHoldOut(Datasplit):
    """ A data split with hold-out set. """

    def __init__(
        self, x_train: np.ndarray, x_test: np.ndarray, x_holdout: np.ndarray,
        y_train: np.ndarray, y_true_train: np.ndarray, y_true_test: np.ndarray,
        holdout_dataset: str,
    ) -> None:
        super().__init__(x_train, x_test, y_train, y_true_train, y_true_test)
        self.x_holdout = x_holdout
        self.holdout_dataset = holdout_dataset
        self.x_holdout = self._transform(self.x_holdout)

    @classmethod
    def from_datasplit(
        cls, datasplit: Datasplit,
        x_holdout: np.ndarray, holdout_dataset: str,
    ) -> "DatasplitHoldOut":
        """ Create from existing data. """

        return cls(
            datasplit.x_train.copy(), datasplit.x_test.copy(),
            x_holdout.copy(), datasplit.y_train.copy(),
            datasplit.y_true_train.copy(), datasplit.y_true_test.copy(),
            holdout_dataset,
        )

    @classmethod
    def from_datasplit_with_holdout_class(
        cls, datasplit: Datasplit, holdout_class: int,
    ) -> "DatasplitHoldOut":
        """ Create from existing data. """

        # Extract hold-out samples
        is_holdout_train = (datasplit.y_true_train == holdout_class).copy()
        is_holdout_test = (datasplit.y_true_test == holdout_class).copy()
        x_holdout = np.vstack([
            datasplit.x_train[is_holdout_train].copy(),
            datasplit.x_test[is_holdout_test].copy(),
        ])

        # Remove class and associated samples from train set
        x_train = datasplit.x_train[~is_holdout_train].copy()
        x_test = datasplit.x_test[~is_holdout_test].copy()
        y_train = datasplit.y_train[~is_holdout_train][
            :, np.arange(datasplit.y_train.shape[1]) != holdout_class
        ].copy()
        y_true_train = np.where(
            datasplit.y_true_train > holdout_class,
            datasplit.y_true_train - 1,
            datasplit.y_true_train,
        )[~is_holdout_train].copy()
        y_true_test = np.where(
            datasplit.y_true_test > holdout_class,
            datasplit.y_true_test - 1,
            datasplit.y_true_test,
        )[~is_holdout_test].copy()
        return DatasplitHoldOut(
            x_train, x_test, x_holdout, y_train,
            y_true_train, y_true_test, f"class{holdout_class}",
        )

    def copy(self) -> "DatasplitHoldOut":
        """ Copies the datasplit.

        Returns:
            DatasplitHoldOut: The copy.
        """

        return DatasplitHoldOut(
            self.x_train.copy(), self.x_test.copy(),
            self.x_holdout.copy(), self.y_train.copy(),
            self.y_true_train.copy(), self.y_true_test.copy(),
            self.holdout_dataset,
        )


class Experiment:
    """ An experiment. """

    def __init__(
        self, dataset_name: str, dataset_kind: str,
        augment_type: str, seed: int, datasplit: Datasplit,
        adv_eps: float,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_kind = dataset_kind
        self.augment_type = augment_type
        self.seed = seed
        self.datasplit = datasplit
        self.adv_eps = adv_eps


def flatten_if_image(inputs: np.ndarray) -> np.ndarray:
    """ Flattens the data if it represents an image. """

    return inputs.reshape(inputs.shape[0], -1).copy()


def get_rl_dataset(dataset_name: str) -> Dataset:
    """ Retrieves a real-world dataset. """

    # Coerce data into dense array
    def coerce(data) -> np.ndarray:
        try:
            return data.toarray()
        except:  # pylint: disable=bare-except
            return data

    # Extract raw data
    raw_mat_data = loadmat(REAL_WORLD_LABEL_TO_PATH[dataset_name])
    x_raw = coerce(raw_mat_data["data"])
    y_partial_raw = coerce(raw_mat_data["partial_target"].transpose())
    y_true_raw = np.argmax(
        coerce(raw_mat_data["target"].transpose()), axis=1)
    classes_in_use = set(range(y_partial_raw.shape[1]))

    # Collect all relevant data
    x_list = []
    y_partial_list = []
    y_true_list: List[int] = []
    mask = np.array(list(sorted(list(classes_in_use))))
    for x_row, y_partial_row, y_true_row in zip(
        x_raw, y_partial_raw, y_true_raw,
    ):
        if int(y_true_row) in classes_in_use:
            x_list.append(x_row)
            y_partial_list.append(y_partial_row[mask])
            y_true_list.append(int(np.where(
                mask == int(y_true_row))[0][0]))
    x_arr = np.array(x_list)
    y_partial_arr = np.array(y_partial_list)
    y_true_arr = np.array(y_true_list)
    x_arr = x_arr[:, x_arr.var(axis=0) > 1e-30].copy()

    # Store dataset
    return Dataset(x_arr, y_partial_arr, y_true_arr)


def get_mnist_dataset(
    dataset_name: str, rng: np.random.Generator, is_ood: bool = False,
) -> Datasplit:
    """ Retrieves an MNIST dataset. """

    # Extract tar file if needed
    if (
        dataset_name == "notmnist" and
        not os.path.exists("external/notMNIST_small")
    ):
        with tarfile.open("external/notMNIST_small.tar.gz", "r:gz") as tar_ref:
            tar_ref.extractall("external")

    # Extract datasets
    if dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root="external", train=True,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
        test_dataset = torchvision.datasets.MNIST(
            root="external", train=False,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
    elif dataset_name == "fmnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root="external", train=True,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="external", train=False,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
    elif dataset_name == "kmnist":
        train_dataset = torchvision.datasets.KMNIST(
            root="external", train=True,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
        test_dataset = torchvision.datasets.KMNIST(
            root="external", train=False,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
    elif dataset_name == "notmnist" and is_ood:
        # Use full set for ood inference
        train_dataset = NotMNIST("external/notMNIST_small")
        test_dataset = NotMNIST("external/notMNIST_small")
    elif dataset_name == "notmnist" and not is_ood:
        # Otherwise, create datasplit
        dataset = NotMNIST("external/notMNIST_small")
        x_full = (dataset.data.view(-1, 1, 28, 28).numpy() / 255).copy()
        y_full_true = dataset.targets.numpy().copy()
        l_classes = np.unique(y_full_true).shape[0]
        y_full = np.zeros((x_full.shape[0], l_classes), dtype=int)
        for i, y_val in enumerate(y_full_true):
            y_full[i, y_val] = 1
        return Dataset(x_full, y_full, y_full_true).create_data_split(rng)
    else:
        raise ValueError()

    # Extract numpy data
    x_train = (train_dataset.data.view(-1, 1, 28, 28).numpy() / 255).copy()
    x_test = (test_dataset.data.view(-1, 1, 28, 28).numpy() / 255).copy()
    y_train_true = train_dataset.targets.numpy().copy()
    y_test_true = test_dataset.targets.numpy().copy()
    l_classes = np.unique(y_train_true).shape[0]
    y_train = np.zeros((x_train.shape[0], l_classes), dtype=int)
    for i, y_val in enumerate(y_train_true):
        y_train[i, y_val] = 1

    # Create dataset
    return Datasplit(
        x_train, x_test, y_train, y_train_true, y_test_true,
    )
