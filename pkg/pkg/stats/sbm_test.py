from collections import namedtuple

import numpy as np
import pandas as pd
from graspologic.utils import remove_loops
from .fisher_exact_nonunity import fisher_exact_nonunity
from scipy.stats import chi2_contingency, combine_pvalues, fisher_exact
from statsmodels.stats.proportion import test_proportions_2indep

SBMResult = namedtuple(
    "sbm_result", ["probabilities", "observed", "possible", "group_counts"]
)


def fit_sbm(A, labels, loops=False):
    if not loops:
        A = remove_loops(A)

    n = A.shape[0]

    node_to_comm_map = dict(zip(np.arange(n), labels))

    # map edges to their incident communities
    source_inds, target_inds = np.nonzero(A)
    comm_mapper = np.vectorize(node_to_comm_map.get)
    source_comm = comm_mapper(source_inds)
    target_comm = comm_mapper(target_inds)

    # get the total number of possible edges for each community -> community cell
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    K = len(unique_labels)

    n_observed = (
        pd.crosstab(
            source_comm,
            target_comm,
            dropna=False,
            rownames=["source"],
            colnames=["target"],
        )
        .reindex(index=unique_labels, columns=unique_labels)
        .fillna(0.0)
    )

    n_possible = np.outer(counts_labels, counts_labels)

    if not loops:
        # then there would have been n fewer possible edges
        n_possible[np.arange(K), np.arange(K)] = (
            n_possible[np.arange(K), np.arange(K)] - counts_labels
        )

    n_possible = pd.DataFrame(
        data=n_possible, index=unique_labels, columns=unique_labels
    )
    n_possible.index.name = "source"
    n_possible.columns.name = "target"

    B_hat = np.divide(n_observed, n_possible)

    counts_labels = pd.Series(index=unique_labels, data=counts_labels)

    return SBMResult(B_hat, n_observed, n_possible, counts_labels)


def binom_2samp(x1, n1, x2, n2, null_odds, method="agresti-caffo"):
    if x1 == 0 or x2 == 0:
        # logging.warn("One or more counts were 0, not running test and returning nan")
        return np.nan, np.nan
    if null_odds != 1 and method != "fisher":
        raise ValueError("Non-unity null odds only works with Fisher's exact test")

    cont_table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    if method == "fisher" and null_odds == 1.0:
        stat, pvalue = fisher_exact(cont_table)
    elif method == "fisher" and null_odds != 1.0:
        stat, pvalue = fisher_exact_nonunity(cont_table, null_odds=null_odds)
    elif method == "chi2":
        stat, pvalue, _, _ = chi2_contingency(cont_table)
    elif method == "agresti-caffo":
        stat, pvalue = test_proportions_2indep(
            x1,
            n1,
            x2,
            n2,
            method="agresti-caffo",
            compare="diff",
            alternative="two-sided",
        )
    else:
        raise ValueError()

    return stat, pvalue


def _make_adjacency_dataframe(data, index):
    df = pd.DataFrame(data=data, index=index.copy(), columns=index.copy())
    df.index.name = "source"
    df.columns.name = "target"
    return df


def stochastic_block_test(
    A1, A2, labels1, labels2, null_odds=1.0, method="fisher_exact"
):

    B1, n_observed1, n_possible1, group_counts1 = fit_sbm(A1, labels1)
    B2, n_observed2, n_possible2, group_counts2 = fit_sbm(A2, labels2)

    # TODO fix this up
    assert n_observed1.index.equals(n_observed2.index)
    assert n_observed1.columns.equals(n_observed2.columns)
    assert n_possible1.index.equals(n_possible2.index)
    assert n_observed1.columns.equals(n_possible2.columns)
    index = n_observed1.index.copy()

    if n_observed1.shape[0] != n_observed2.shape[0]:
        raise ValueError()

    K = n_observed1.shape[0]

    uncorrected_pvalues = np.empty((K, K), dtype=float)
    uncorrected_pvalues = _make_adjacency_dataframe(uncorrected_pvalues, index)

    stats = np.empty((K, K), dtype=float)
    stats = _make_adjacency_dataframe(stats, index)

    for i in index:
        for j in index:
            curr_stat, curr_pvalue = binom_2samp(
                n_observed1.loc[i, j],
                n_possible1.loc[i, j],
                n_observed2.loc[i, j],
                n_possible2.loc[i, j],
                method=method,
                null_odds=null_odds,
            )
            uncorrected_pvalues.loc[i, j] = curr_pvalue
            stats.loc[i, j] = curr_stat

    misc = {}
    misc["uncorrected_pvalues"] = uncorrected_pvalues
    misc["stats"] = stats
    misc["probabilities1"] = B1
    misc["probabilities2"] = B2
    misc["observed1"] = n_observed1
    misc["obserbed2"] = n_observed2
    misc["possible1"] = n_possible1
    misc["possible2"] = n_possible2
    misc["group_counts1"] = group_counts1
    misc["group_counts2"] = group_counts2
    misc["null_odds"] = null_odds

    # TODO how else to combine pvalues
    run_pvalues = uncorrected_pvalues.values
    run_pvalues = run_pvalues[~np.isnan(run_pvalues)]
    stat, pvalue = combine_pvalues(run_pvalues, method="fisher")
    n_tests = len(run_pvalues)
    misc["n_tests"] = n_tests
    return stat, pvalue, misc
