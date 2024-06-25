""" Script to create tables and plots. """

import itertools
import math
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import ecdf, ks_2samp, ttest_rel
from sklearn.metrics.pairwise import rbf_kernel

algos_to_idx = {
    # ML methods
    "pl-knn-2005": 0,
    "pl-svm-2008": 1,
    "ipal-2015": 2,
    "pl-ecoc-2017": 3,
    # Deep Learning methods
    "proden-2020": 4,
    "rc-2020": 5,
    "cc-2020": 6,
    "valen-2021": 7,
    "cavl-2021": 8,
    "pop-2023": 9,
    "crosel-2024": 10,
    "dst-pll-2024": 11,
    # Proden variations
    "proden-l2": 12,
    "proden-dropout": 13,
    "proden-ens": 14,
    "proden-adv-ens": 15,
    "proden-edl": 16,
    # Our methods
    "robust-pll": 17,
    "robust-pll-ens": 18,
}
idx_to_algos = {v: k for k, v in algos_to_idx.items()}
datasets_to_idx = {
    # Real-world datasets
    "bird-song": 0,
    "lost": 1,
    "mir-flickr": 2,
    "msrc-v2": 3,
    "soccer": 4,
    "yahoo-news": 5,
    # MNIST datasets
    "mnist": 6,
    "fmnist": 7,
    "kmnist": 8,
    "notmnist": 9,
}
idx_to_datasets = {v: k for k, v in datasets_to_idx.items()}
idx_to_split = {
    0: "train",
    1: "test",
    2: "ood",
}
algo_sort = {
    # Non-ensemble methods
    "pl-knn-2005": 0,
    "pl-svm-2008": 1,
    "ipal-2015": 2,
    "pl-ecoc-2017": 3,
    "proden-2020": 4,
    "proden-l2": 5,
    "proden-edl": 6,
    "rc-2020": 7,
    "cc-2020": 8,
    "valen-2021": 9,
    "cavl-2021": 10,
    "pop-2023": 11,
    "crosel-2024": 12,
    "dst-pll-2024": 13,
    "robust-pll": 14,
    # Ensemble methods
    "proden-dropout": 15,
    "proden-ens": 16,
    "proden-adv-ens": 17,
    "robust-pll-ens": 18,
}
algo_type = {
    # Non-ensemble methods
    "pl-knn-2005": 0,
    "pl-svm-2008": 0,
    "ipal-2015": 0,
    "pl-ecoc-2017": 0,
    "proden-2020": 0,
    "proden-l2": 0,
    "proden-edl": 0,
    "rc-2020": 0,
    "cc-2020": 0,
    "valen-2021": 0,
    "cavl-2021": 0,
    "pop-2023": 0,
    "crosel-2024": 0,
    "dst-pll-2024": 0,
    "robust-pll": 0,
    # Ensemble methods
    "proden-dropout": 1,
    "proden-ens": 1,
    "proden-adv-ens": 1,
    "robust-pll-ens": 1,
}
uses_nn = {
    # Non-ensemble methods
    "pl-knn-2005": 0,
    "pl-svm-2008": 0,
    "ipal-2015": 0,
    "pl-ecoc-2017": 0,
    "proden-2020": 1,
    "proden-l2": 1,
    "proden-edl": 1,
    "rc-2020": 1,
    "cc-2020": 1,
    "valen-2021": 1,
    "cavl-2021": 1,
    "pop-2023": 1,
    "crosel-2024": 1,
    "dst-pll-2024": 0,
    "robust-pll": 1,
    # Ensemble methods
    "proden-dropout": 2,
    "proden-ens": 2,
    "proden-adv-ens": 2,
    "robust-pll-ens": 2,
}
algo_displaynames = {
    # Non-ensemble methods
    "pl-knn-2005": "\\textsc{PlKnn} (2005)",
    "pl-svm-2008": "\\textsc{PlSvm} (2008)",
    "ipal-2015": "\\textsc{Ipal} (2015)",
    "pl-ecoc-2017": "\\textsc{PlEcoc} (2017)",
    "proden-2020": "\\textsc{Proden} (2020)",
    "proden-l2": "\\textsc{Proden+L2}",
    "proden-edl": "\\textsc{Proden+Edl}",
    "rc-2020": "\\textsc{Rc} (2020)",
    "cc-2020": "\\textsc{Cc} (2020)",
    "valen-2021": "\\textsc{Valen} (2021)",
    "cavl-2021": "\\textsc{Cavl} (2022)",
    "pop-2023": "\\textsc{Pop} (2023)",
    "crosel-2024": "\\textsc{CroSel} (2024)",
    "dst-pll-2024": "\\textsc{DstPll} (2024)",
    "robust-pll": "\\textsc{RobustPll} (ours)",
    # Ensemble methods
    # "proden-rf": "\\textsc{Proden+Rf}",
    "proden-dropout": "\\textsc{Proden+Dropout}",
    "proden-ens": "\\textsc{Proden+Ens}",
    "proden-adv-ens": "\\textsc{Proden+AdvEns}",
    "robust-pll-ens": "\\textsc{RobustPll+Ens} (ours)",
}
datasetkind_to_idx = {
    "rl": 0,
    "mnistlike": 1,
}
idx_to_datasetkind = {v: k for k, v in datasetkind_to_idx.items()}
augtype_to_idx = {
    "rl": 0,
    "instance-dependent": 1,
}
idx_to_augtype = {v: k for k, v in augtype_to_idx.items()}


def mmd2(
    values1: np.ndarray, values2: np.ndarray,
) -> float:
    """ Compute MMD. """

    values1 = values1.reshape(-1, 1)
    values2 = values2.reshape(-1, 1)
    mat11 = rbf_kernel(values1, values1, 1.0)
    mat22 = rbf_kernel(values2, values2, 1.0)
    mat12 = rbf_kernel(values1, values2, 1.0)
    h = mat11 + mat22 - 2 * mat12
    np.fill_diagonal(h, 0)

    # Compute MMD
    res = float(h.sum() / (h.shape[0] * (h.shape[0] - 1)))
    return math.sqrt(res if res > 0 else 0.0)


@njit(cache=True, parallel=True)
def mmd(
    values1: np.ndarray, values2: np.ndarray, rbf_gamma: float,
) -> float:
    """ Compute MMD. """

    n = int(values1.shape[0])
    mat_sum = 0.0
    for i in prange(n):
        for j in prange(n):
            if i == j:
                continue
            mat_sum += math.exp(
                -rbf_gamma * (values1[i] - values1[j]) ** 2)

    for i in prange(n):
        for j in prange(n):
            if i == j:
                continue
            mat_sum += math.exp(
                -rbf_gamma * (values2[i] - values2[j]) ** 2)

    for i in prange(n):
        for j in prange(n):
            if i == j:
                continue
            mat_sum -= 2 * math.exp(
                -rbf_gamma * (values1[i] - values2[j]) ** 2)

    # Compute MMD
    res = mat_sum / (n * (n - 1))
    return math.sqrt(res if res > 0 else 0.0)


def compute_area(x1, y1, x2, y2) -> float:
    """ Computes the area between plots. """

    y1_interp = np.interp(x2, x1, y1)
    area = float(np.trapezoid(y1_interp - y2, x=x2))
    return area


def get_acc_std_per_algo_and_mnist_like_dataset(dataset: int) -> pd.DataFrame:
    """ Get stats per algo and MNIST-like dataset. """

    conn = sqlite3.connect("results/all_res.db")
    data_algo_seed_df = pd.read_sql_query((
        "select * from results where "
        f"dataset = {dataset} and "
        "datasetkind = 1 and "
        "(ooddataset = 6 or (dataset = 6 and ooddataset = 9) "
        "or ooddataset = -1) and "
        "augmenttype = 1 and adveps = 0 and split = 1"
    ), conn).groupby(by=[
        "dataset", "algo", "seed",
    ], as_index=False)["correct"].mean()
    conn.close()

    res = []
    max_alg = [0, 0]
    max_acc = [0.0, 0.0]
    max_subset = [None, None]
    for algo in idx_to_algos:
        subset = data_algo_seed_df.query(f"algo == {algo}")["correct"]
        at = algo_type[idx_to_algos[algo]]
        if np.mean(subset) > max_acc[at]:
            max_acc[at] = np.mean(subset)
            max_alg[at] = algo
            max_subset[at] = subset
    for algo in idx_to_algos:
        subset = data_algo_seed_df.query(f"algo == {algo}")["correct"]
        if algo not in max_alg:
            test = ttest_rel(subset, max_subset[algo_type[idx_to_algos[algo]]])
            assert hasattr(test, "pvalue")
            is_significant = getattr(test, "pvalue") < 0.05
        else:
            is_significant = False
        res.append([
            idx_to_algos[algo], np.mean(subset), np.std(subset),
            algo_type[idx_to_algos[algo]], is_significant,
            algo_sort[idx_to_algos[algo]],
        ])

    return pd.DataFrame(
        res, columns=["algo", "mean", "std", "algotype", "sig_diff", "sort"]
    ).sort_values(by=["sort"])


def get_acc_std_per_algo_and_datasetkind(
    datasetkind_name: str, augmenttype_name: str,
) -> pd.DataFrame:
    """ Get a DataFrame with accuracies on the given dataset kind. """

    datasetkind = datasetkind_to_idx[datasetkind_name]
    augmenttype = augtype_to_idx[augmenttype_name]

    conn = sqlite3.connect("results/all_res.db")
    out_df = pd.read_sql_query((
        "select * from results where "
        f"datasetkind = {datasetkind} and "
        "(ooddataset = 6 or "
        "(dataset = 6 and ooddataset = 9) or ooddataset = -1) and "
        f"augmenttype = {augmenttype} and adveps = 0 and split = 1"
    ), conn).groupby(by=[
        "dataset", "datasetkind", "ooddataset", "adveps",
        "algo", "seed", "augmenttype", "split",
    ], as_index=False)["correct"].mean()
    conn.close()

    res = []
    max_alg = [0, 0]
    max_acc = [0.0, 0.0]
    max_subset = [None, None]
    for algo in idx_to_algos:
        subset = out_df.query(f"algo == {algo}")["correct"]
        at = algo_type[idx_to_algos[algo]]
        if np.mean(subset) > max_acc[at]:
            max_acc[at] = np.mean(subset)
            max_alg[at] = algo
            max_subset[at] = subset
    for algo in idx_to_algos:
        subset = out_df.query(f"algo == {algo}")["correct"]
        if algo not in max_alg:
            test = ttest_rel(subset, max_subset[algo_type[idx_to_algos[algo]]])
            assert hasattr(test, "pvalue")
            is_significant = getattr(test, "pvalue") < 0.05
        else:
            is_significant = False
        res.append([
            idx_to_algos[algo], np.mean(subset), np.std(subset),
            algo_type[idx_to_algos[algo]], is_significant,
            algo_sort[idx_to_algos[algo]],
        ])

    return pd.DataFrame(
        res, columns=["algo", "mean", "std", "algotype", "sig_diff", "sort"]
    ).sort_values(by=["sort"])


def fts(f: float, b: bool = False, desired_width: int = 4) -> str:
    """ Converts a float to a table entry. """

    res = f"{f * 100:.1f}"
    if len(res) == desired_width - 1:
        res = "\\phantom{0}" + res
    if b:
        res = "\\textbf{" + res + "}"
    return res


def print_acc_table() -> None:
    """ Print accuracy table. """

    mnist = get_acc_std_per_algo_and_mnist_like_dataset(6)
    fmnist = get_acc_std_per_algo_and_mnist_like_dataset(7)
    kmnist = get_acc_std_per_algo_and_mnist_like_dataset(8)
    notmnist = get_acc_std_per_algo_and_mnist_like_dataset(9)
    rr = get_acc_std_per_algo_and_datasetkind("rl", "rl")

    merged_df = pd.concat([
        mnist.loc[:, ["algo", "mean", "std", "sig_diff"]],
        fmnist.loc[:, ["mean", "std", "sig_diff"]],
        kmnist.loc[:, ["mean", "std", "sig_diff"]],
        notmnist.loc[:, ["mean", "std", "sig_diff"]],
        rr.loc[:, ["mean", "std", "sig_diff"]],
    ], axis=1)

    print()
    print("\\midrule")
    for _, row in merged_df.iterrows():
        if row.iloc[0] == "proden-dropout":
            print("\\midrule")
        row_strs = [f"{algo_displaynames[row.iloc[0]]} & "]
        for i in [1, 4, 7, 10, 13]:
            row_strs.append(
                f"{fts(row.iloc[i], not row.iloc[i + 2])} "
                f"($\\pm$ {fts(row.iloc[i + 1], desired_width=3)})"
            )
            if i != 13:
                row_strs.append(" & ")
            else:
                row_strs.append(" \\\\")
        row_str = "".join(row_strs)
        print(row_str)
    print("\\bottomrule")
    print()


def create_rl_adv_table(algos):
    """ Creates the adversarial experiment results. """

    table = []
    adveps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    conn = sqlite3.connect("results/all_res.db")
    subset_res = pd.read_sql_query((
        "select * from results where dataset >= 0 and dataset <= 5 "
        "and ooddataset = -1 and augmenttype = 0 and split = 1"
    ), conn)
    conn.close()

    # Get best algo per adveps
    current_best_acc = {f"{e:.1f}": 0.0 for e in adveps_list}
    current_best_alg = {f"{e:.1f}": 0 for e in adveps_list}
    current_best_data = {f"{e:.1f}": [0.0] for e in adveps_list}
    for algo in algos:
        row = [algo]
        for adveps in adveps_list:
            adv_res = []
            for dataset in range(6):
                for seed in range(5):
                    res = subset_res.query(
                        f"dataset == {dataset} and "
                        f"algo == {algos_to_idx[algo]} and "
                        f"adveps == {adveps} and seed == {seed}"
                    )
                    adv_res.append(float(res['correct'].mean()))
            res_mean = float(np.mean(adv_res))
            if res_mean > current_best_acc[f"{adveps:.1f}"]:
                current_best_acc[f"{adveps:.1f}"] = res_mean
                current_best_alg[f"{adveps:.1f}"] = algos_to_idx[algo]
                current_best_data[f"{adveps:.1f}"] = adv_res

    # Create table
    for algo in algos:
        row = [algo]
        for adveps in adveps_list:
            adv_res = []
            for dataset in range(6):
                for seed in range(5):
                    res = subset_res.query(
                        f"dataset == {dataset} and "
                        f"algo == {algos_to_idx[algo]} and "
                        f"adveps == {adveps} and seed == {seed}"
                    )
                    adv_res.append(float(res['correct'].mean()))
            res_mean = np.mean(adv_res)
            res_std = np.std(adv_res)

            if current_best_alg[f"{adveps:.1f}"] != algos_to_idx[algo]:
                best_data = current_best_data[f"{adveps:.1f}"]
                test_res = ttest_rel(adv_res, best_data)
                assert hasattr(test_res, "pvalue")
                sig_diff = float(getattr(test_res, "pvalue")) < 0.05
            else:
                sig_diff = False

            row.append(res_mean)
            row.append(res_std)
            row.append(sig_diff)
        table.append(row)
    return pd.DataFrame(
        table, columns=sum(
            ([["algo"]] + [[
                f"adv_{adveps:.1f}_acc",
                f"adv_{adveps:.1f}_std",
                f"adv_{adveps:.1f}_sig_diff",
            ] for adveps in adveps_list]),
            []
        )
    )


def print_adv_table():
    """ Print adversarial experiments table. """

    non_ensemble_nn_algos = [algo for algo, i in uses_nn.items() if i == 1]
    ensemble_nn_algos = [algo for algo, i in uses_nn.items() if i == 2]
    res1 = create_rl_adv_table(non_ensemble_nn_algos)
    res2 = create_rl_adv_table(ensemble_nn_algos)
    res = pd.concat([res1, res2])

    print()
    print("\\midrule")
    for _, row in res.iterrows():
        if row.iloc[0] == "proden-dropout":
            print("\\midrule")
        row_strs = [f"{algo_displaynames[row.iloc[0]]} & "]
        for i in [1, 4, 7, 10, 13, 16]:
            row_strs.append(
                f"{fts(row.iloc[i], not row.iloc[i + 2])} "
                f"($\\pm$ {fts(row.iloc[i + 1], desired_width=3)})"
            )
            if i != 16:
                row_strs.append(" & ")
            else:
                row_strs.append(" \\\\")
        row_str = "".join(row_strs)
        print(row_str)
    print("\\bottomrule")
    print()


def fts_entr(f: float, b: bool = False) -> str:
    """ Converts a float to a table entry. """

    res = f"{f:.4f}"
    if b:
        res = "\\textbf{" + res + "}"
    return res


def print_ood_table():
    """ Print table with OOD results. """

    conn = sqlite3.connect("results/all_res.db")
    subset_res = pd.read_sql_query((
        "select * from results where "
        "dataset = 6 and ooddataset = 9 and "
        "datasetkind = 1 and augmenttype = 1 and "
        "adveps = 0 and split != 0"
    ), conn)
    conn.close()

    # Compute parameter of kernel: Avg. distance
    dists = 0.0
    count = 10000
    rng = np.random.Generator(np.random.PCG64(42))
    for inst1, inst2 in rng.choice(
        subset_res.shape[0], size=(10000, 2), replace=True
    ):
        dists += float(np.abs(
            subset_res.loc[inst1, "entropy"] -
            subset_res.loc[inst2, "entropy"]
        ))
    rbf_gamma = 1 / (2 * ((dists / count) ** 2))

    # Get best algo
    current_best_ks = {t: 0.0 for t in itertools.product(
        [0, 1], ["area", "ks", "mmd"])}
    current_best_alg = {t: 0 for t in itertools.product(
        [0, 1], ["area", "ks", "mmd"])}
    for algo in algo_sort:
        for testset in ["mnist"]:
            for oodset in ["notmnist"]:
                if testset == oodset:
                    continue

                test_entr = subset_res.query(
                    f"algo == {algos_to_idx[algo]} and "
                    f"dataset == {datasets_to_idx[testset]} and "
                    f"ooddataset == {datasets_to_idx[oodset]} and "
                    "split == 1"
                )["entropy"]
                ood_entr = subset_res.query(
                    f"algo == {algos_to_idx[algo]} and "
                    f"dataset == {datasets_to_idx[testset]} and "
                    f"ooddataset == {datasets_to_idx[oodset]} and "
                    "split == 2"
                )["entropy"]

                # KS test statistic
                kstest_res = ks_2samp(test_entr, ood_entr)
                key = (algo_type[algo], "ks")
                stat = getattr(kstest_res, "statistic") \
                    * getattr(kstest_res, "statistic_sign")
                if stat > current_best_ks[key]:
                    current_best_ks[key] = stat
                    current_best_alg[key] = algos_to_idx[algo]

                # Area between cdfs
                test_ecdf = ecdf(test_entr).cdf
                ood_ecdf = ecdf(ood_entr).cdf
                key = (algo_type[algo], "area")
                stat = compute_area(
                    test_ecdf.quantiles, test_ecdf.probabilities,
                    ood_ecdf.quantiles, ood_ecdf.probabilities,
                )
                if stat > current_best_ks[key]:
                    current_best_ks[key] = stat
                    current_best_alg[key] = algos_to_idx[algo]

                # MMD between test-ood
                key = (algo_type[algo], "mmd")
                mmd_sign = (
                    1 if (
                        np.mean(ood_entr) - np.mean(test_entr)
                    ) > 0 else -1
                )
                stat = mmd_sign * mmd(
                    test_entr.values, ood_entr.values, rbf_gamma)
                if stat > current_best_ks[key]:
                    current_best_ks[key] = stat
                    current_best_alg[key] = algos_to_idx[algo]

    # Print table
    print()
    print("\\midrule")
    for algo in algo_sort:
        if algo == "proden-dropout":
            print("\\midrule")

        row_strs = [f"{algo_displaynames[algo]} & "]
        for testset in ["mnist"]:
            for oodset in ["notmnist"]:
                if testset == oodset:
                    continue

                test_entr = subset_res.query(
                    f"algo == {algos_to_idx[algo]} and "
                    f"dataset == {datasets_to_idx[testset]} and "
                    f"ooddataset == {datasets_to_idx[oodset]} and "
                    "split == 1"
                )["entropy"]
                ood_entr = subset_res.query(
                    f"algo == {algos_to_idx[algo]} and "
                    f"dataset == {datasets_to_idx[testset]} and "
                    f"ooddataset == {datasets_to_idx[oodset]} and "
                    "split == 2"
                )["entropy"]

                # KS test statistic
                kstest_res = ks_2samp(test_entr, ood_entr)
                ks_stat = getattr(kstest_res, "statistic")
                ks_sign = getattr(kstest_res, "statistic_sign")
                ks_sign_txt = "\\phantom{-}" if ks_sign > 0 else "-"
                test_ecdf = ecdf(test_entr).cdf
                ood_ecdf = ecdf(ood_entr).cdf
                area_stat = compute_area(
                    test_ecdf.quantiles, test_ecdf.probabilities,
                    ood_ecdf.quantiles, ood_ecdf.probabilities,
                )
                area_sign = 1 if area_stat > 0 else -1
                area_stat = np.abs(area_stat)
                area_sign_txt = "\\phantom{-}" if area_sign > 0 else "-"
                mmd_sign = (
                    1 if (
                        np.mean(ood_entr) - np.mean(test_entr)
                    ) > 0 else -1
                )
                mmd_sign_txt = "\\phantom{-}" if mmd_sign > 0 else "-"
                mmd_stat = mmd(test_entr.values, ood_entr.values, rbf_gamma)

                # Create line
                is_area_bold = (
                    algos_to_idx[algo] == current_best_alg[
                        (algo_type[algo], "area")]
                )
                is_ks_bold = (
                    algos_to_idx[algo] == current_best_alg[
                        (algo_type[algo], "ks")]
                )
                is_mmd_bold = (
                    algos_to_idx[algo] == current_best_alg[
                        (algo_type[algo], "mmd")]
                )
                row_strs.append(area_sign_txt)
                row_strs.append(fts_entr(area_stat, is_area_bold))
                row_strs.append(" & ")
                row_strs.append(ks_sign_txt)
                row_strs.append(fts_entr(ks_stat, is_ks_bold))
                row_strs.append(" & ")
                row_strs.append(mmd_sign_txt)
                row_strs.append(fts_entr(mmd_stat, is_mmd_bold))
                row_strs.append(" \\\\")
        print("".join(row_strs))
    print("\\bottomrule")
    print()


def plot_entropy_cdf():
    """ Plots the entropy CDFs. """

    conn = sqlite3.connect("results/all_res.db")
    subset_res = pd.read_sql_query((
        "select * from results where "
        "dataset = 6 and ooddataset = 9 and "
        "augmenttype = 1 and adveps = 0 and split != 0"
    ), conn)
    conn.close()

    fs = 9
    # column_width = 3.3249219444  # inch
    text_width = 7.02625  # inch

    latex_preamble = """
\\usepackage[utf8]{inputenc}
\\usepackage{microtype}
\\usepackage{amsmath}
    """
    # \\renewcommand{\\rmdefault}{ptm}
    # \\renewcommand{\\sfdefault}{phv}

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": latex_preamble,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "font.size": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "savefig.bbox": "tight",
    })

    linestyle_fmts = [
        (0, (1, 1)),  (0, (5, 5)), (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)), (0, ()), (0, (3, 1, 1, 1)),
        (0, (3, 5, 1, 5)), (0, (5, 3)), (0, (3, 1, 3, 1, 1, 1)),
    ]

    _, axis = plt.subplots(1, 2, figsize=(text_width - 1, 1.8))
    plt.subplots_adjust(wspace=0.4)

    axis[0].grid(alpha=1.0, color="#e7e7e7")
    axis[1].grid(alpha=1.0, color="#e7e7e7")

    cmap = plt.get_cmap("tab20")

    algo_display = {
        "proden-2020": "\\textsc{Proden}",
        "proden-edl": "\\textsc{P+Edl}",
        "dst-pll-2024": "\\textsc{DstPll}",
        "robust-pll": "Our method",
        "proden-dropout": "\\textsc{P+Dropout}",
        "proden-ens": "\\textsc{P+Ens}",
        "proden-adv-ens": "\\textsc{P+AdvEns}",
        "robust-pll-ens": "Ours+\\textsc{Ens}",
    }

    ci = 0
    for algo in [
        "proden-2020", "proden-edl", "dst-pll-2024", "robust-pll",
    ]:
        test_res = subset_res.query(
            f"algo == {algos_to_idx[algo]} and split == 1"
        )["entropy"]
        ood_res = subset_res.query(
            f"algo == {algos_to_idx[algo]} and split == 2"
        )["entropy"]

        test_ecdf = ecdf(test_res).cdf
        ood_ecdf = ecdf(ood_res).cdf

        axis[0].plot(
            test_ecdf.quantiles, test_ecdf.probabilities,
            linestyle=linestyle_fmts[ci // 2],
            label=f"{algo_display[algo]}", color=cmap(ci),
        )
        ci += 1
        axis[0].plot(
            ood_ecdf.quantiles, ood_ecdf.probabilities,
            linestyle=linestyle_fmts[ci // 2],
            label="_nolegend_", color=cmap(ci),
        )
        ci += 1
        axis[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
        axis[0].set_xlim(0, 1)
        axis[0].set_ylim(0, 1)
        axis[0].set_xlabel("Predictive Entropy (Test\\,/\\,OOD)", fontsize=fs)
        axis[0].set_ylabel("Empirical CDF", fontsize=fs)

    ci = 0
    for algo in [
        "proden-dropout", "proden-ens", "proden-adv-ens", "robust-pll-ens",
    ]:
        test_res = subset_res.query(
            f"algo == {algos_to_idx[algo]} and split == 1"
        )["entropy"]
        ood_res = subset_res.query(
            f"algo == {algos_to_idx[algo]} and split == 2"
        )["entropy"]

        test_ecdf = ecdf(test_res).cdf
        ood_ecdf = ecdf(ood_res).cdf

        axis[1].plot(
            test_ecdf.quantiles, test_ecdf.probabilities,
            linestyle=linestyle_fmts[ci // 2],
            label=f"{algo_display[algo]}", color=cmap(ci),
        )
        ci += 1
        axis[1].plot(
            ood_ecdf.quantiles, ood_ecdf.probabilities,
            linestyle=linestyle_fmts[ci // 2],
            label="_nolegend_", color=cmap(ci),
        )
        ci += 1
        axis[1].legend(loc="lower center",
                       bbox_to_anchor=(0.5, 1.01), ncol=2)
        axis[1].set_xlim(0, 1)
        axis[1].set_ylim(0, 1)
        axis[1].set_xlabel(
            "Predictive Entropy (Test\\,/\\,OOD)", fontsize=fs)
        axis[1].set_ylabel("Empirical CDF", fontsize=fs)

    plt.savefig("plots/entropy_cdfs.pdf")


if __name__ == "__main__":
    print_acc_table()
    print_adv_table()
    print_ood_table()
    plot_entropy_cdf()
