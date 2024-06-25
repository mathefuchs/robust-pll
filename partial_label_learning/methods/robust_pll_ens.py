""" Module for Evidential Deep Learning for PLL. """

from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed

from partial_label_learning.methods.robust_pll import RobustPll
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


def _run_model(
    seed: int, inputs: np.ndarray, partial_targets: np.ndarray,
    coef: float, adv_eps: float,
) -> RobustPll:
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)
    model = RobustPll(rng, False, adv_eps)
    model.fit_with_max_coeff(inputs, partial_targets, coef)
    return model


class RobustPllEnsemble(PllBaseClassifier):
    """
    Robust PLL.
    """

    def __init__(
        self, rng: np.random.Generator,
        debug: bool = False, adv_eps: float = 0.0,
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.adv_eps = adv_eps
        self.models: List[RobustPll] = []

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

        res = Parallel(n_jobs=5)(
            delayed(_run_model)(
                self.rng.choice(1000), inputs,
                partial_targets, coef, self.adv_eps,
            )
            for coef in [0.5, 1.0, 2.0, 4.0, 8.0]
        )
        self.models = res  # type: ignore
        prob, unc = self.models[0].w_probs, self.models[0].w_unc
        if prob is None or unc is None:
            raise ValueError()
        for model in self.models[1:]:
            prob1, unc1 = model.w_probs, model.w_unc
            if prob1 is None or unc1 is None:
                raise ValueError()
            prob += prob1
            unc += unc1
        prob /= 5
        unc /= 5
        return SplitResult.from_scores_with_uncertainty(
            self.rng, prob, unc,
        )

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        for m in self.models:
            m.predict(inputs)

        prob, unc = self.models[0].w_probs, self.models[0].w_unc
        if prob is None or unc is None:
            raise ValueError()
        for model in self.models[1:]:
            prob1, unc1 = model.w_probs, model.w_unc
            if prob1 is None or unc1 is None:
                raise ValueError()
            prob += prob1
            unc += unc1
        prob /= 5
        unc /= 5
        return SplitResult.from_scores_with_uncertainty(
            self.rng, prob, unc,
        )
