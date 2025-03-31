""" Dataset overview """

from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from partial_label_learning.data import Experiment

all_exps = {}
meta_info = {}
for fname in tqdm(sorted(glob("./experiments/*.pt"))):
    exp: Experiment = torch.load(fname, weights_only=False)
    key = exp.dataset_name
    meta_info[key] = (
        exp.datasplit.x_train.shape[0] + exp.datasplit.x_test.shape[0],
        exp.datasplit.x_train.shape[1],
        exp.datasplit.y_train.shape[1],
    )
    if key not in all_exps:
        all_exps[key] = []
    all_exps[key].append(float(np.sum(exp.datasplit.y_train, axis=1).mean()))

for k, vs in all_exps.items():
    v = float(np.mean(vs))
    print(k)
    print(meta_info[k])
    print(f"Avg. candidates: {v:.3f}")
    print()
