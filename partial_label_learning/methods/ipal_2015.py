""" Module for IPAL. """

from typing import List, Optional

import cvxpy as cp
import numpy as np
from scipy.sparse import lil_array
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class Ipal(PllBaseClassifier):
    """
    IPAL by Zhang and Yu,
    "Solving the Partial Label Learning Problem: An Instance-Based Approach."
    """

    def __init__(
        self,
        rng: np.random.Generator,
        debug: bool = False,
        adv_eps: float = 0.0,
        k_neighbors: int = 10,
        alpha: float = 0.95,
        max_iterations: int = 100,
    ) -> None:
        self.rng = rng
        self.adv_eps = adv_eps
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.knn: Optional[NearestNeighbors] = None
        self.weight_matrix = None
        self.initial_confidence_matrix = None
        self.final_confidence_matrix = None
        self.inst_feats: Optional[cp.Parameter] = None
        self.neighbor_feats: Optional[cp.Parameter] = None
        self.weight_vars: Optional[cp.Variable] = None
        self.prob: Optional[cp.Problem] = None
        self.x_train: Optional[np.ndarray] = None
        self.num_classes = None
        self.train_res: Optional[SplitResult] = None

    def _solve_neighbor_weights_prob(
        self, inst_feats: np.ndarray, inst_neighbors: np.ndarray,
    ) -> np.ndarray:
        # Formulate optimization problem
        assert self.inst_feats is not None
        assert self.neighbor_feats is not None
        assert self.weight_vars is not None
        assert self.prob is not None
        assert self.x_train is not None
        self.inst_feats.value = inst_feats
        self.neighbor_feats.value = np.vstack([
            self.x_train[j] for j in inst_neighbors
        ])
        self.prob.solve(solver=cp.MOSEK)

        # Return weights
        if self.prob.status != "optimal":
            raise ValueError("Failed to find weights.")
        return self.weight_vars.value

    def fit(self, inputs: np.ndarray, partial_targets: np.ndarray) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        # Compute nearest neighbors
        inputs = flatten_if_image(inputs)
        self.x_train = inputs
        num_insts = inputs.shape[0]
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=-1)
        self.knn.fit(inputs)
        self.weight_matrix = lil_array((num_insts, num_insts), dtype=float)
        self.initial_confidence_matrix: Optional[np.ndarray] = None
        self.final_confidence_matrix: Optional[np.ndarray] = None

        # Neighborhood weight optimization problem
        num_feats = inputs.shape[1]
        self.inst_feats = cp.Parameter(num_feats)
        self.neighbor_feats = cp.Parameter((self.k_neighbors, num_feats))
        self.weight_vars = cp.Variable(self.k_neighbors)
        constraints = [self.weight_vars >= 0]
        cost = cp.sum_squares(
            self.inst_feats - self.neighbor_feats.T @ self.weight_vars)
        self.prob = cp.Problem(cp.Minimize(cost), constraints)  # type: ignore

        # Compute neighbors for each instance
        nn_indices: np.ndarray = self.knn.kneighbors(
            return_distance=False)  # type: ignore

        # Solve optimization problem to find weights
        for inst, inst_neighbors in self.loop_wrapper(enumerate(nn_indices)):
            # Formulate optimization problem
            weight_vars = self._solve_neighbor_weights_prob(
                self.x_train[inst], inst_neighbors)

            # Store resulting weights
            for neighbor_idx, weight in zip(inst_neighbors, weight_vars):
                if float(weight) > 1e-10:
                    self.weight_matrix[neighbor_idx, inst] = float(weight)

        # Compact information and normalize
        self.weight_matrix = self.weight_matrix.tocoo()
        norm = self.weight_matrix.sum(axis=0)
        self.weight_matrix /= np.where(norm > 1e-10, norm, 1)

        # Initial labeling confidence
        self.num_classes = partial_targets.shape[1]
        initial_labeling_conf = np.zeros((num_insts, self.num_classes))
        for inst in range(num_insts):
            count_labels = np.count_nonzero(partial_targets[inst, :])
            initial_labeling_conf[inst, partial_targets[inst, :] == 1] = \
                1 / count_labels

        # Iterative propagation
        curr_labeling_conf = initial_labeling_conf.copy()
        for _ in self.loop_wrapper(range(self.max_iterations)):
            # Propagation
            curr_labeling_conf = (
                self.alpha * self.weight_matrix.T @ curr_labeling_conf +
                (1 - self.alpha) * initial_labeling_conf
            )

            # Rescaling
            for inst in range(num_insts):
                sum_labels = np.sum(
                    curr_labeling_conf[inst, partial_targets[inst, :] == 1])
                curr_labeling_conf[inst, :] = np.where(
                    partial_targets[inst, :] == 1,
                    curr_labeling_conf[inst, :] / sum_labels,
                    0.0
                )

        # Set confidence matrices
        self.initial_confidence_matrix = initial_labeling_conf
        self.final_confidence_matrix = curr_labeling_conf

        # Compute class probability masses
        initial_class_mass: np.ndarray = np.sum(
            self.initial_confidence_matrix, axis=0)
        final_class_mass: np.ndarray = np.sum(
            self.final_confidence_matrix, axis=0)

        # Correct for imbalanced class masses
        scores = self.final_confidence_matrix.copy()
        for class_lbl in range(self.num_classes):
            if final_class_mass[class_lbl] > 1e-10:
                scores[:, class_lbl] *= initial_class_mass[class_lbl] / \
                    final_class_mass[class_lbl]

        # Return predictions
        self.train_res = SplitResult.from_scores(self.rng, scores)
        return self.train_res

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.final_confidence_matrix is None or \
                self.initial_confidence_matrix is None:
            raise ValueError("Fit must be called before predict.")

        # Solve optimization problem to find weights
        assert self.knn is not None
        assert self.num_classes is not None
        assert self.train_res is not None
        assert self.x_train is not None
        inputs = flatten_if_image(inputs)
        nn_indices = self.knn.kneighbors(inputs, return_distance=False)
        scores_list: List[List[float]] = []
        for test_inst, train_inst_neighbors in self.loop_wrapper(
            enumerate(nn_indices)
        ):
            # Formulate optimization problem
            weight_vars = self._solve_neighbor_weights_prob(
                inputs[test_inst, :], train_inst_neighbors)

            # Use resulting weights
            scores_list.append([])
            for class_lbl in range(self.num_classes):
                class_vector = inputs[test_inst, :].copy()
                for train_neighbor_idx, train_neighbor_weight in zip(
                    train_inst_neighbors, weight_vars
                ):
                    if class_lbl == self.train_res.pred[train_neighbor_idx] and \
                            float(train_neighbor_weight) > 1e-10:
                        class_vector -= train_neighbor_weight * \
                            self.x_train[train_neighbor_idx]
                scores_list[-1].append(float(
                    np.dot(class_vector, class_vector)))

        # Return predictions
        prob = np.array(scores_list)
        prob = np.max(prob, axis=1, keepdims=True) - prob
        return SplitResult.from_logits(self.rng, prob)
