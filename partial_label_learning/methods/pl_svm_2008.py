""" Module for PL-SVM. """

import math
from typing import Optional

import numpy as np
from numba import njit

from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class WeightVector:
    """ Wraps a weight vector. """

    def __init__(self, m_features: int, l_classes: int) -> None:
        self.m_features = m_features
        self.l_classes = l_classes
        self.n_dims = m_features * l_classes
        self.weights = np.zeros(self.n_dims, dtype=float)

    def norm(self) -> float:
        """ Returns the norm of the weight vector.

        Returns:
            float: The norm.
        """

        return float(np.linalg.norm(self.weights, 2))

    def scale(self, scale: float) -> None:
        """ Scales the vector.

        Args:
            scale (float): Scales the vector.
        """

        self.weights *= scale

    def add_phi_xy(self, scale: float, x_i: np.ndarray, y_i: int) -> None:
        """ Add a multiple of Phi(x, y) to the weight vector.

        Args:
            scale (float): The scale.
            x_i (np.ndarray): The features.
            y_i (int): The candidate label.
        """

        self.weights[
            self.m_features * y_i:self.m_features * (y_i + 1)
        ] += scale * x_i


@njit(cache=True, parallel=False)
def _wt_phi_xy(
    weights: np.ndarray, x_i: np.ndarray, y_i: int, m_features: int,
) -> float:
    """ Computes w^T * Phi(x, y).

    Args:
        x_i (np.ndarray): The features.
        y_i (int): The candidate label.

    Returns:
        float: The result.
    """

    return float(np.sum(
        weights[m_features * y_i:m_features * (y_i + 1)]
        * x_i
    ))


class PlSvm(PllBaseClassifier):
    """
    PL-SVM by Nguyen and Caruana,
    "Classification with Partial Labels."
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
        lambda_reg: float = 1.0,
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.adv_eps = adv_eps
        self.lambda_reg = lambda_reg
        self.num_classes = None

        # Model
        self.weight_vector: Optional[WeightVector] = None

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
        self.num_classes = partial_targets.shape[1]

        # Init weight vector
        self.weight_vector = WeightVector(
            inputs.shape[1], self.num_classes)
        num_insts = inputs.shape[0]

        # Stochastic training loop
        max_iterations = max(100000, 10 * num_insts)
        for epoch in range(max_iterations):
            # Pick random element
            inst = self.rng.choice(num_insts)
            x_i = inputs[inst]
            ys_i = partial_targets[inst]

            # Compute max margin
            pos_scores = [
                _wt_phi_xy(
                    self.weight_vector.weights, x_i,
                    class_lbl, self.weight_vector.m_features,
                )
                if ys_i[class_lbl] == 1 else -np.inf
                for class_lbl in range(self.num_classes)
            ]
            max_pos_class = int(np.argmax(pos_scores))
            neg_scores = [
                _wt_phi_xy(
                    self.weight_vector.weights, x_i,
                    class_lbl, self.weight_vector.m_features,
                )
                if ys_i[class_lbl] == 0 else -np.inf
                for class_lbl in range(self.num_classes)
            ]
            max_neg_class = int(np.argmax(neg_scores))

            # Compute eta
            eta = 1 / (self.lambda_reg * (epoch + 1))
            weight_scaling = max(1e-9, 1 - eta * self.lambda_reg)

            # Regularize weight
            self.weight_vector.scale(weight_scaling)

            # Add feedback from violations
            if pos_scores[max_pos_class] - neg_scores[max_neg_class] < 1:
                self.weight_vector.add_phi_xy(eta, x_i, max_pos_class)
                self.weight_vector.add_phi_xy(-eta, x_i, max_neg_class)

            # Normalize vector
            w_norm = self.weight_vector.norm()
            if w_norm > 1e-10:
                projection = 1 / (math.sqrt(self.lambda_reg) * w_norm)
                if projection < 1:
                    self.weight_vector.scale(projection)

        # Return predictions
        return self._predict_internal(inputs, partial_targets, True)

    def _predict_internal(
        self, data: np.ndarray, candidates: Optional[np.ndarray],
        is_train: bool,
    ) -> SplitResult:
        if not self.weight_vector:
            raise ValueError()
        if data.shape[0] == 0:
            raise ValueError()
        assert self.num_classes is not None
        scores = -np.inf * np.ones((data.shape[0], self.num_classes))
        for i, x_i in enumerate(data):
            for j in range(self.num_classes):
                # If in transductive setting, only assign
                # valid score for candidate labels
                if (
                    not is_train or
                    (candidates is not None and candidates[i, j] == 1)
                ):
                    scores[i, j] = _wt_phi_xy(
                        self.weight_vector.weights, x_i,
                        j, self.weight_vector.m_features,
                    )

        # Return predictions
        return SplitResult.from_logits(self.rng, scores)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        inputs = flatten_if_image(inputs)
        return self._predict_internal(inputs, None, False)
