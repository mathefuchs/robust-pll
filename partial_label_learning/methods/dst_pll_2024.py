""" Module for DST-PLL. """

import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult
from reference_models.vae import VariationalAutoEncoder


def yager_combine(
    m_bpas: List[Dict[int, float]],
    universal_set: int,
    prune_prob: float = 1e-10,
) -> Dict[int, float]:
    """ Yager rule of combination.

    Args:
        m_bpas (List[Dict[int, float]]): The evidences to combine.
        universal_set (int): The universal set in this context.
        prune_prob (float): Prune probabilities smaller than this.
        Defaults to 1e-10.

    Returns:
        Dict[int, float]: The combined evidence.
    """

    # Combine observations
    curr_m_bpa = {universal_set: 1.0}
    for next_m_bpa in m_bpas:
        new_m_bpa = {}
        for set1, set1_prob in curr_m_bpa.items():
            for set2, set2_prob in next_m_bpa.items():
                intersect = set1 & set2
                prob_prod = set1_prob * set2_prob
                # Increase probability of intersecting evidence
                if intersect not in new_m_bpa:
                    new_m_bpa[intersect] = prob_prod
                else:
                    new_m_bpa[intersect] += prob_prod
        curr_m_bpa = new_m_bpa

    # Assign probability of empty set to universal set
    if 0 in curr_m_bpa:
        if universal_set not in curr_m_bpa:
            curr_m_bpa[universal_set] = curr_m_bpa[0]
        else:
            curr_m_bpa[universal_set] += curr_m_bpa[0]
        del curr_m_bpa[0]

    # Prune too small probabilities
    for subset in list(curr_m_bpa.keys()):
        if curr_m_bpa[subset] < prune_prob:
            del curr_m_bpa[subset]

    # Rescale
    sum_probs = sum(curr_m_bpa.values())
    if sum_probs != 1.0:
        for subset in curr_m_bpa:
            curr_m_bpa[subset] /= sum_probs

    return curr_m_bpa


class CandidateLabelsEncoder:
    """ Class for encoding candidate label sets. """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def encode_candidate_list(
        self, candidates: Union[np.ndarray, List[int]],
    ) -> int:
        """ Encodes a candidate list into an integer.

        Args:
            candidates (Union[np.ndarray, List[int]]): The binary candidate vector.

        Returns:
            int: The encoded candidate list.
        """

        return sum(
            (1 << code_pos) * int(candidates[code_pos])
            for code_pos in range(self.num_classes)
        )

    def decode_candidate_list(self, encoded_candidates: int) -> List[int]:
        """ Decodes an encoded candidate list back to a list.

        Args:
            encoded_candidates (int): The encoded representation.

        Returns:
            List[int]: The binary class list.
        """

        return [
            1
            if ((1 << class_lbl) & encoded_candidates) != 0
            else 0
            for class_lbl in range(self.num_classes)
        ]


class DstPll(PllBaseClassifier):
    """
    Dempster-Shafer Theory for Partial Label Learning.
    """

    def __init__(
        self, rng: np.random.Generator, debug: bool = False,
        adv_eps: float = 0.0, dataset_kind: str = "uci",
        dataset_name: str = "",
    ) -> None:
        self.rng = rng
        self.debug = debug
        self.adv_eps = adv_eps
        self.num_classes: Optional[int] = None
        self.knn: Optional[NearestNeighbors] = None
        self.label_encoder: Optional[CandidateLabelsEncoder] = None
        self.y_train: Optional[np.ndarray] = None

        # Init variational auto-encoder
        if dataset_kind == "mnistlike":
            self.k_neighbors = 20
            if torch.cuda.is_available():
                cuda_idx = random.randrange(torch.cuda.device_count())
                self.device = torch.device(f"cuda:{cuda_idx}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.vae = VariationalAutoEncoder()
            self.vae.load_state_dict(torch.load(
                f"./saved_models/vae-{dataset_name}.pt",
                map_location=self.device,
            ))
            self.vae.eval()
            self.vae.to(self.device)
        else:
            self.k_neighbors = 10
            self.device = None
            self.vae = None

    def _encode_data(self, inputs: np.ndarray) -> np.ndarray:
        """ Encode the input data. """

        inputs = flatten_if_image(inputs)
        if self.vae is None or self.device is None:
            return inputs

        with torch.no_grad():
            x_tensor = torch.tensor(
                inputs, dtype=torch.float32, device=self.device)
            _, x_enc, _ = self.vae(x_tensor, compute_loss=False)
        return x_enc.cpu().numpy()

    def _infer_labeling(self, nn_indices, is_train: bool) -> SplitResult:
        """ Infer labeling by combining evidence from neighbors. """

        # Encode candidate lists as bit strings;
        # Python integers have arbitrary bit length
        assert self.num_classes is not None
        assert self.label_encoder is not None
        assert self.y_train is not None
        num_classes_mask = (1 << self.num_classes) - 1
        train_targets_enc = [
            self.label_encoder.encode_candidate_list(y_row)
            for y_row in self.y_train
        ]

        # Combine evidence from nearest neighbors
        # using Dempster's rule of combination
        bel_list: List[List[float]] = []
        unc_list: List[float] = []
        bel_pl_diff_list: List[float] = []
        reject_list: List[bool] = []
        all_bpas: List[Dict[int, float]] = []
        for inst, train_inst_neighbors in enumerate(nn_indices):
            # We are sure that the answer is from the given candidate set
            inst_candidates = (
                train_targets_enc[inst]
                if is_train else num_classes_mask
            )
            evidence_to_combine = [{inst_candidates: 1.0}]

            # Combine evidence from neighbors using Yager's rule
            for train_neighbor_idx in map(int, train_inst_neighbors):
                # Retrieve candidates of neighbor
                neighbor_candidates = train_targets_enc[train_neighbor_idx]

                # If all probability mass is already allotted
                # to candidates in evidence or evidence is disjoint,
                # do not use it since it has no influence
                if inst_candidates & neighbor_candidates \
                        in (0, inst_candidates):
                    continue

                # Append evidence
                evidence_to_combine.append({
                    (neighbor_candidates & inst_candidates): 0.5,
                    inst_candidates: 0.5,
                })

            # Combine evidence
            m_bpa = yager_combine(evidence_to_combine, inst_candidates)
            all_bpas.append(m_bpa)

            # Extract all single-item beliefs
            all_single_beliefs: List[float] = []
            curr_class_idx = 1
            for _ in range(self.num_classes):
                all_single_beliefs.append(m_bpa.get(curr_class_idx, 0.0))
                curr_class_idx <<= 1
            unc = 1 - sum(all_single_beliefs)
            curr_belief_max, curr_belief_max_idx = max(
                (v, i) for i, v in enumerate(all_single_beliefs))
            bel_list.append(all_single_beliefs)
            unc_list.append(unc)

            # Evaluate reject option
            still_plausible_num = 0
            curr_class_idx = 1
            all_single_plau = []
            for _ in range(self.num_classes):
                plausibility = sum(
                    subset_val for subset, subset_val in m_bpa.items()
                    if (subset & curr_class_idx) == curr_class_idx
                )
                all_single_plau.append(plausibility)
                if plausibility >= curr_belief_max:
                    still_plausible_num += 1
                curr_class_idx <<= 1
            reject_list.append(still_plausible_num != 1)

            # Belief-Plausibility difference
            max_bel_pl_diff = curr_belief_max - np.max(
                np.array(all_single_plau)[
                    np.arange(len(all_single_plau)) != curr_belief_max_idx
                ]
            )
            bel_pl_diff_list.append(max_bel_pl_diff)

        # Extract predictions
        bel_arr = np.array(bel_list)
        unc_arr = np.array(unc_list)
        reject_arr = np.array(bel_pl_diff_list)
        prob_arr = bel_arr.copy()
        high_uncertainty = unc_arr == 1
        for i in range(nn_indices.shape[0]):
            # If uncertainty close to one, determine subset with
            # highest mass and smallest cardinality
            if high_uncertainty[i]:
                most_likely_subset_encoded = sorted(
                    list(all_bpas[i].items()),
                    key=lambda t: (-t[1], t[0].bit_count()),
                )[0][0]
                mask = np.array(self.label_encoder.decode_candidate_list(
                    most_likely_subset_encoded))
                prob_arr[i] = mask / np.count_nonzero(mask)

        return SplitResult.from_scores_with_reject(
            self.rng, prob_arr, reject_arr)

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

        # Compute nearest neighbors
        inputs = self._encode_data(inputs)
        self.num_classes = partial_targets.shape[1]
        self.y_train = partial_targets
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=-1)
        self.knn.fit(inputs)

        # Label encoder
        self.label_encoder = CandidateLabelsEncoder(self.num_classes)

        # Compute nearest neighbors for each instance
        nn_indices = self.knn.kneighbors(return_distance=False)

        # Return train predictions
        return self._infer_labeling(nn_indices, True)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        # Compute nearest neighbors for each instance
        assert self.knn is not None
        inputs = self._encode_data(inputs)
        nn_indices = self.knn.kneighbors(
            inputs, return_distance=False)

        # Return test predictions
        return self._infer_labeling(nn_indices, False)
