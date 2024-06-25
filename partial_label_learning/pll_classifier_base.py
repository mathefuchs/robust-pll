""" Module for base PLL classifier. """

from abc import ABC, abstractmethod

import numpy as np

from partial_label_learning.result import SplitResult


class PllBaseClassifier(ABC):
    """ Base PLL classifier. """

    @abstractmethod
    def __init__(
        self, rng: np.random.Generator, debug: bool, adv_eps: float,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
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

        raise NotImplementedError()

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        raise NotImplementedError()
