""" Create data for PLL experiments. """

import random
import string
from typing import Union

import numpy as np
import torch
from joblib import Parallel, delayed

from partial_label_learning.config import AUG_TYPE, SELECTED_DATASETS
from partial_label_learning.data import (DatasplitHoldOut, Experiment,
                                         get_mnist_dataset, get_rl_dataset)

DEBUG = False


def create_experiment_data(
    dataset_name: str, dataset_kind: str, augment_type: str, seed: int,
    adv_eps: float, ood_dataset_name: Union[None, str, int] = None,
):
    """ Create experiment data. """

    # Init random generator
    torch.manual_seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Load dataset
    if dataset_kind == "rl":
        dataset = get_rl_dataset(dataset_name)
        datasplit = dataset.create_data_split(rng)
    elif dataset_kind == "mnistlike":
        datasplit = get_mnist_dataset(dataset_name, rng)
    else:
        raise ValueError()

    # Augment dataset
    if augment_type == "uniform":
        datasplit = datasplit.augment_targets(
            rng=rng, r_candidates=2, percent_partially_labeled=0.5,
            eps_cooccurrence=0.0,
        )
    elif augment_type == "class-dependent":
        datasplit = datasplit.augment_targets(rng, 1, 0.7, 0.7)
    elif augment_type == "instance-dependent":
        datasplit = datasplit.augment_targets_instance_dependent(rng)

    # Add optional ood data
    if ood_dataset_name is not None:
        if isinstance(ood_dataset_name, str):
            if dataset_kind == "rl":
                ood_dataset = get_rl_dataset(ood_dataset_name)
                datasplit = DatasplitHoldOut.from_datasplit(
                    datasplit, ood_dataset.x_full, ood_dataset_name)
            elif dataset_kind == "mnistlike":
                ood_datasplit = get_mnist_dataset(ood_dataset_name, rng, True)
                datasplit = DatasplitHoldOut.from_datasplit(
                    datasplit, ood_datasplit.x_test, ood_dataset_name)
            else:
                raise ValueError()
        elif isinstance(ood_dataset_name, int):
            datasplit = DatasplitHoldOut.from_datasplit_with_holdout_class(
                datasplit, ood_dataset_name)
        else:
            raise ValueError()

    # Save experiment
    exp = Experiment(
        dataset_name, dataset_kind, augment_type, seed, datasplit, adv_eps)
    if not DEBUG:
        fname = "".join([
            random.choice(string.ascii_lowercase) for _ in range(10)])
        torch.save(exp, f"./experiments/{fname}.pt")
    else:
        avg_cl = np.mean(np.count_nonzero(datasplit.y_train, axis=1))
        print()
        print(f"Dataset: {dataset_name}")
        print(f"Avg. #CL: {avg_cl:.4f}")
        torch.save(exp, f"exp{seed}.pt")


if __name__ == "__main__":
    if not DEBUG:
        # Download all data
        d_rng = np.random.Generator(np.random.PCG64(42))
        get_mnist_dataset("mnist", d_rng)
        get_mnist_dataset("fmnist", d_rng)
        get_mnist_dataset("kmnist", d_rng)
        get_mnist_dataset("notmnist", d_rng)

        # Create experiment data
        Parallel(n_jobs=12)(
            delayed(create_experiment_data)(
                dataset_name, dataset_kind, augment_type,
                seed, adv_eps, ood_dataset_name,
            )
            # All datasets used
            for dataset_name, (_, dataset_kind) in SELECTED_DATASETS.items()
            # All partial-label noise types
            for augment_type in AUG_TYPE
            # Do not add noise for real-world datasets
            if (
                (dataset_kind == "rl" and augment_type == "rl") or
                (dataset_kind != "rl" and augment_type != "rl")
            )
            # All out-of-distribtution datasets to test
            for ood_dataset_name in [None, "notmnist"]
            if (
                (dataset_name == "mnist" and ood_dataset_name == "notmnist")
                or (dataset_name != "mnist" and ood_dataset_name is None)
            )
            # Use adversarial perturbations on real-world datasets
            for adv_eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            if adv_eps == 0.0 or augment_type == "rl"
            # Repeat 5 times for reporting averages and stds
            for seed in range(5)
        )
    else:
        for s in range(5):
            # Create single data for debugging
            create_experiment_data(
                "lost", "rl", "rl", s, 0.0)
