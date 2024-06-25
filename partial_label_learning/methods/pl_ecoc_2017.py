""" Module for PL-ECOC. """

import math
from typing import List, Set

import numpy as np
from sklearn.svm import LinearSVC
from tqdm import tqdm

from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class PlEcoc(PllBaseClassifier):
    """
    PL-ECOC by Zhang, Yu, and Tang,
    "Disambiguation-Free Partial Label Learning."
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
    ) -> None:
        self.rng = rng
        self.adv_eps = adv_eps
        self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.num_insts = None
        self.num_classes = None
        self.codeword_length = None
        self.enc_multiplier: List[int] = []
        self.y_train_enc = None
        self.num_classes_mask = None
        self.min_binary_train_size = None

        # Model
        self.model_is_fit = False
        self.coding_matrix: np.ndarray = np.array([])
        self.perf_matrix: np.ndarray = np.array([])
        self.binary_clfs: List[LinearSVC] = []

    def _compute_coding_column(self) -> bool:
        assert self.num_classes is not None
        assert self.num_classes_mask is not None
        assert self.y_train_enc is not None
        assert self.min_binary_train_size is not None
        assert self.codeword_length is not None

        # If less then 10 classes, exhaustive search
        possible_codes = []
        if self.num_classes <= 10:
            for pos_set in range(1, (self.num_classes_mask >> 1) + 1):
                # Define positive and negative class labels
                neg_set = self.num_classes_mask - pos_set

                # Count training instances
                num_pos = 0
                num_neg = 0
                for cand in self.y_train_enc:
                    if (cand & pos_set) == cand:
                        num_pos += 1
                    elif (cand & neg_set) == cand:
                        num_neg += 1

                # Check if code provides valid dichotomy
                if (
                    num_pos and num_neg and
                    num_pos + num_neg >= self.min_binary_train_size
                ):
                    possible_codes.append(pos_set)

        # Else, random choice
        else:
            max_retries = 10000
            retries = 0
            already_used: Set[int] = set()

            while (
                retries < max_retries and
                len(possible_codes) < self.codeword_length
            ):
                # Define positive and negative class labels
                pos_set_list = list(map(int, list(self.rng.choice(
                    2, size=self.num_classes-1, replace=True))))
                pos_set_list.append(0)
                if all(item == 0 for item in pos_set_list):
                    retries += 1
                    continue
                pos_set = sum(
                    self.enc_multiplier[i] * pos_set_list[i]
                    for i in range(self.num_classes)
                )

                if pos_set in already_used:
                    retries += 1
                    continue
                neg_set = self.num_classes_mask - pos_set

                # Count training instances
                num_pos = 0
                num_neg = 0
                for cand in self.y_train_enc:
                    if (cand & pos_set) == cand:
                        num_pos += 1
                    elif (cand & neg_set) == cand:
                        num_neg += 1

                # Check if code provides valid dichotomy
                if (
                    num_pos and num_neg and
                    num_pos + num_neg >= self.min_binary_train_size
                ):
                    retries = 0
                    possible_codes.append(pos_set)
                    already_used.add(pos_set)
                else:
                    retries += 1

        # Sample from possible codes
        final_codeword_length = min(self.codeword_length, len(possible_codes))
        if not final_codeword_length:
            return False
        codes_picked: List[int] = sorted(map(int, list(self.rng.choice(
            possible_codes, size=final_codeword_length,
            replace=False, shuffle=False,
        ))))

        # Extract coding matrix
        self.coding_matrix = self.coding_matrix[
            :self.num_classes, :len(codes_picked)
        ]
        self.codeword_length = len(codes_picked)
        for i, code in enumerate(codes_picked):
            for j, bit in enumerate(list(reversed(f"{code:b}"))):
                self.coding_matrix[j, i] = 1 if bit == "1" else -1
        return True

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
        self.num_insts = inputs.shape[0]
        self.num_classes = partial_targets.shape[1]
        self.codeword_length = int(math.ceil(10 * np.log2(self.num_classes)))

        # Encoded candidates
        self.enc_multiplier = [
            2 ** code_pos for code_pos in range(self.num_classes)
        ]
        self.y_train_enc = [
            sum(
                self.enc_multiplier[code_pos] * int(y_row[code_pos])
                for code_pos in range(self.num_classes)
            ) for y_row in partial_targets
        ]
        self.num_classes_mask = sum(self.enc_multiplier)
        self.min_binary_train_size = max(1, self.num_insts // 10)

        # Compute coding matrix
        self.coding_matrix = -np.ones(
            (self.num_classes, self.codeword_length), dtype=int)
        self.binary_clfs = []
        if not self._compute_coding_column():
            # Unable to build dichotomies
            return SplitResult.from_scores(
                self.rng, np.ones_like(partial_targets, dtype=float))

        for codeword_idx in self.loop_wrapper(range(self.codeword_length)):
            # q-bits column coding
            column_coding = self.coding_matrix[:, codeword_idx]

            # Derive training sets
            pos_set = int(np.sum(
                np.where(column_coding == 1, 1, 0) * self.enc_multiplier))
            neg_set = self.num_classes_mask - pos_set
            data_mask = np.zeros(self.num_insts, dtype=bool)
            contains_positive = False
            contains_negative = False
            targets = []
            for i, cand in enumerate(self.y_train_enc):
                if (cand & pos_set) == cand:
                    data_mask[i] = True
                    targets.append(True)
                    contains_positive = True
                elif (cand & neg_set) == cand:
                    data_mask[i] = True
                    targets.append(False)
                    contains_negative = True

            # Found dichotomy with enough training instances
            if contains_positive and contains_negative:
                # Train binary classifier on dichotomy
                clf = LinearSVC(
                    random_state=self.rng.integers(int(1e6)),
                    loss="squared_hinge", max_iter=10000, dual="auto",  # type: ignore
                )
                clf.fit(inputs[data_mask, :], targets)
                self.binary_clfs.append(clf)
            else:
                # Invalid state
                raise RuntimeError("Invalid state.")

        # Pre-compute binary classifier predictions
        pred_results = [
            self.binary_clfs[codeword_idx].predict(inputs)
            for codeword_idx in range(self.codeword_length)
        ]

        # Compute performance matrix
        self.perf_matrix = np.zeros(
            (self.num_classes, self.codeword_length), dtype=float)
        for class_idx in range(self.num_classes):
            for codeword_idx in range(self.codeword_length):
                mask = partial_targets[:, class_idx] == 1
                mask_norm = np.count_nonzero(mask)
                self.perf_matrix[class_idx, codeword_idx] = (
                    np.count_nonzero(
                        pred_results[codeword_idx][mask]
                        == (self.coding_matrix[class_idx, codeword_idx] == 1)
                    )
                    / (mask_norm if mask_norm != 0 else 1)
                )

        # Normalize performance matrix
        norm = self.perf_matrix.sum(axis=1)
        self.perf_matrix = (
            self.perf_matrix.transpose() / np.where(norm < 1e-6, 1, norm)
        ).transpose()
        self.model_is_fit = True

        # Return predictions
        return self._predict_internal(inputs, partial_targets, True)

    def _predict_internal(
        self, data: np.ndarray, candidates: np.ndarray, is_train: bool,
    ) -> SplitResult:
        assert self.num_classes is not None
        if not self.model_is_fit:
            return SplitResult.from_scores(
                self.rng, np.ones((data.shape[0], self.num_classes)))
        if data.shape[0] == 0:
            raise ValueError()
        assert self.codeword_length is not None

        # Precompute all decision function outputs
        decision_func_outputs = np.vstack([
            self.binary_clfs[codeword_idx].decision_function(data)
            for codeword_idx in range(self.codeword_length)
        ])

        scores_list: List[np.ndarray] = []
        for inst in range(data.shape[0]):
            class_scores = np.zeros(self.num_classes)
            for class_idx in range(self.num_classes):
                if is_train and candidates[inst, class_idx] != 1:
                    continue
                for codeword_idx in range(self.codeword_length):
                    perf_entry = self.perf_matrix[class_idx, codeword_idx]
                    if perf_entry == 0.0:
                        continue
                    pred = float(decision_func_outputs[codeword_idx, inst])
                    coding_entry = self.coding_matrix[class_idx, codeword_idx]
                    class_scores[class_idx] += (
                        perf_entry * np.exp(pred * coding_entry)
                    )
            scores_list.append(class_scores)

        # Return predictions
        return SplitResult.from_scores(self.rng, np.array(scores_list))

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        inputs = flatten_if_image(inputs)
        return self._predict_internal(inputs, np.array([]), False)
