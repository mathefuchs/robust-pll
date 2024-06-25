""" Script to run all experiments. """

import io
import random
import string
from glob import glob
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import scipy
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from partial_label_learning.config import (AUG_TYPE, DATASET_KIND,
                                           SELECTED_DATASETS)
from partial_label_learning.data import DatasplitHoldOut, Experiment
from partial_label_learning.methods.cavl_2021 import Cavl
from partial_label_learning.methods.cc_2020 import CC
from partial_label_learning.methods.crosel_2024 import CroSel
from partial_label_learning.methods.dst_pll_2024 import DstPll
from partial_label_learning.methods.ipal_2015 import Ipal
from partial_label_learning.methods.pl_ecoc_2017 import PlEcoc
from partial_label_learning.methods.pl_knn_2005 import PlKnn
from partial_label_learning.methods.pl_svm_2008 import PlSvm
from partial_label_learning.methods.pop_2023 import Pop
from partial_label_learning.methods.proden_2020 import Proden
from partial_label_learning.methods.proden_adv_ens import ProdenAdvEns
from partial_label_learning.methods.proden_dropout import ProdenDropout
from partial_label_learning.methods.proden_edl import ProdenEdl
from partial_label_learning.methods.proden_ens import ProdenEnsemble
from partial_label_learning.methods.proden_l2 import ProdenL2
from partial_label_learning.methods.rc_2020 import RC
from partial_label_learning.methods.robust_pll import RobustPll
from partial_label_learning.methods.robust_pll_ens import RobustPllEnsemble
from partial_label_learning.methods.valen_2021 import Valen
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import Result

DEBUG = False
ALGOS: Dict[str, Tuple[int, Type[PllBaseClassifier]]] = {
    # ML methods
    "pl-knn-2005": (0, PlKnn),
    "pl-svm-2008": (1, PlSvm),
    "ipal-2015": (2, Ipal),
    "pl-ecoc-2017": (3, PlEcoc),
    # Deep Learning methods
    "proden-2020": (4, Proden),
    "rc-2020": (5, RC),
    "cc-2020": (6, CC),
    "valen-2021": (7, Valen),
    "cavl-2021": (8, Cavl),
    "pop-2023": (9, Pop),
    "crosel-2024": (10, CroSel),
    "dst-pll-2024": (11, DstPll),
    # Proden variations
    "proden-l2": (12, ProdenL2),
    "proden-dropout": (13, ProdenDropout),
    "proden-ens": (14, ProdenEnsemble),
    "proden-adv-ens": (15, ProdenAdvEns),
    "proden-edl": (16, ProdenEdl),
    # Our methods
    "robust-pll": (17, RobustPll),
    "robust-pll-ens": (18, RobustPllEnsemble),
}


def area_between_cdfs(entr1: np.ndarray, entr2: np.ndarray) -> float:
    """ Return the area between two entropy cdfs. """

    cdf1 = scipy.stats.ecdf(entr1).cdf
    cdf2 = scipy.stats.ecdf(entr2).cdf
    x1 = cdf1.quantiles
    y1 = cdf1.probabilities
    x2 = cdf2.quantiles
    y2 = cdf2.probabilities
    y1_interp = np.interp(x2, x1, y1)
    area = float(np.trapz(y1_interp - y2, x=x2))
    return area


def fts(number: float, max_digits: int = 6) -> str:
    """ Float to string. """

    return f"{float(number):.{max_digits}f}".rstrip("0").rstrip(".")


def get_header() -> str:
    """ Builds the header. """

    return "dataset,datasetkind,ooddataset,adveps,algo,seed,augmenttype," + \
        "split,truelabel,predlabel,correct,guess,conf,entropy\n"


def append_output(
    output: List[str], algo_name: str, exp: Experiment,
    result: Result, split: int,
) -> None:
    """ Create output from result. """

    if split == 0:
        res = result.train_result
        true_label_list = exp.datasplit.y_true_train
    elif split == 1:
        res = result.test_result
        true_label_list = exp.datasplit.y_true_test
    elif split == 2:
        res = result.holdout_result
        true_label_list = -np.ones_like(exp.datasplit.y_true_test)
        if res is None:
            raise ValueError()
    else:
        raise ValueError()

    for true_label, pred, conf, entr, is_guess in zip(
        true_label_list, res.pred, res.conf, res.entr, res.is_guessing,
    ):
        output.append(f"{int(SELECTED_DATASETS[exp.dataset_name][0])}")
        output.append(f",{int(DATASET_KIND[exp.dataset_kind])}")
        if isinstance(exp.datasplit, DatasplitHoldOut):
            ood_dataset = exp.datasplit.holdout_dataset
            if ood_dataset.startswith("class"):
                ood_dataset_idx = -int(ood_dataset[5:]) - 2
            else:
                ood_dataset_idx = int(SELECTED_DATASETS[ood_dataset][0])
            output.append(f",{ood_dataset_idx},{fts(exp.adv_eps)}")
        else:
            output.append(f",-1,{fts(exp.adv_eps)}")
        output.append(f",{int(ALGOS[algo_name][0])},{int(exp.seed)}")
        output.append(f",{int(AUG_TYPE[exp.augment_type])},{int(split)}")
        output.append(f",{int(true_label)},{int(pred)},")
        output.append(f"{int(true_label == pred)},{int(is_guess)},")
        output.append(f"{fts(conf)},{fts(entr)}\n")


def print_debug_msg(
    algo_name: str, exp: Experiment, result: Result,
) -> None:
    """ Print debug message. """

    train_acc = accuracy_score(
        exp.datasplit.y_true_train, result.train_result.pred)
    test_acc = accuracy_score(
        exp.datasplit.y_true_test, result.test_result.pred)
    if isinstance(exp.datasplit, DatasplitHoldOut):
        avg_test_entr = np.median(result.test_result.entr)
        avg_ood_entr = np.median(result.get_holdout_result().entr)
        area_entr = area_between_cdfs(
            result.test_result.entr,
            result.get_holdout_result().entr)
    else:
        avg_test_entr = np.median(result.test_result.entr)
        avg_ood_entr = 0.0
        area_entr = 0.0
    print(", ".join([
        f"{exp.dataset_name: >20}", f"{algo_name: >15}",
        f"{exp.augment_type: >18}", f"{exp.seed}",
        f"{train_acc:.3f}", f"{test_acc:.3f}",
        f"{result.train_result.frac_guessing():.3f}",
        f"{result.test_result.frac_guessing():.3f}",
        f"{avg_test_entr:.3f}",
        f"{avg_ood_entr:.3f}",
        f"{area_entr:.3f}",
    ]))


def run_experiment(fname: str, algo_name: str, algo_type: Type[PllBaseClassifier]) -> None:
    """ Runs the given experiment. """

    # Run experiment
    exp: Experiment = torch.load(fname)
    rng = np.random.Generator(np.random.PCG64(exp.seed))
    if issubclass(algo_type, DstPll):
        algo = DstPll(
            rng, False, exp.adv_eps,
            exp.dataset_kind, exp.dataset_name,
        )
    else:
        algo = algo_type(rng, False, exp.adv_eps)
    result = Result(
        train_result=algo.fit(
            exp.datasplit.x_train,
            exp.datasplit.y_train,
        ),
        test_result=algo.predict(exp.datasplit.x_test),
        holdout_result=(
            algo.predict(exp.datasplit.x_holdout)
            if isinstance(exp.datasplit, DatasplitHoldOut)
            else None
        ),
    )

    output = [get_header()]
    append_output(output, algo_name, exp, result, split=0)
    append_output(output, algo_name, exp, result, split=1)
    if result.holdout_result is not None:
        append_output(output, algo_name, exp, result, split=2)

    if DEBUG:
        # Print debug message
        print_debug_msg(algo_name, exp, result)
        csv_df = pd.read_csv(io.StringIO("".join(output)))
        csv_df.to_parquet(
            f"results/result_{algo_name}_{exp.seed}.parquet.gz",
            compression="gzip",
        )
    else:
        # Store predictions
        fname = "".join([
            random.choice(string.ascii_lowercase) for _ in range(12)])
        csv_df = pd.read_csv(io.StringIO("".join(output)))
        csv_df.to_parquet(
            f"results/all/{fname}.parquet.gz", compression="gzip")


if __name__ == "__main__":
    if not DEBUG:
        # Run all experimental settings
        Parallel(n_jobs=6)(
            delayed(run_experiment)(fname, algo_name, algo_type)
            for fname in tqdm(list(sorted(glob("./experiments/*.pt"))))
            for algo_name, (_, algo_type) in ALGOS.items()
        )
    else:
        # Run single experiments for debugging
        for s in range(5):
            run_experiment(f"exp{s}.pt", "robust-pll", RobustPll)
