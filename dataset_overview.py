""" Dataset overview """

from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from partial_label_learning.data import Experiment

avg_cands = {}
meta_info = {}
frac_ground_truth = {}
for fname in tqdm(sorted(glob("./experiments/*.pt"))):
    exp: Experiment = torch.load(fname, weights_only=False)
    key = exp.dataset_name
    meta_info[key] = (
        exp.datasplit.x_train.shape[0] + exp.datasplit.x_test.shape[0],
        exp.datasplit.x_train.shape[1:],
        exp.datasplit.y_train.shape[1],
    )
    if key not in avg_cands:
        avg_cands[key] = []
    if key not in frac_ground_truth:
        frac_ground_truth[key] = []
    avg_cands[key].append(float(np.sum(exp.datasplit.y_train, axis=1).mean()))
    frac_ground_truth[key].append(float(
        np.count_nonzero(np.sum(exp.datasplit.y_train, axis=1) == 1)
        / exp.datasplit.y_train.shape[0]
    ))

for k, vs in avg_cands.items():
    v = float(np.mean(vs))
    fg = float(np.mean(frac_ground_truth[k]))
    print(k)
    print(meta_info[k])
    print(f"Avg. candidates: {v:.3f}")
    print(f"Fraction with ground-truth: {fg:.3f}")
    print()
