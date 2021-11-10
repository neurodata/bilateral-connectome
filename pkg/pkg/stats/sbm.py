from collections import namedtuple

import numpy as np
from numpy.lib.utils import source
import pandas as pd
from graspologic.utils import remove_loops
from scipy.stats import combine_pvalues

from .binomial import binom_2samp, binom_2samp_paired

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


def offdiag_indices_from(arr):
    upper_rows, upper_cols = np.triu_indices_from(arr, k=1)
    lower_rows, lower_cols = np.tril_indices_from(arr, k=-1)
    rows = np.concatenate((upper_rows, lower_rows))
    cols = np.concatenate((upper_cols, lower_cols))
    return rows, cols


def stochastic_block_test_paired(A1, A2, labels):
    index, group_indices, group_counts = np.unique(
        labels, return_counts=True, return_inverse=True
    )

    K = len(index)

    empty = np.empty((K, K), dtype=float)
    uncorrected_pvalues = _make_adjacency_dataframe(empty.copy(), index)
    stats = _make_adjacency_dataframe(empty.copy(), index)
    empty = np.empty((K, K), dtype=int)
    both = _make_adjacency_dataframe(empty.copy(), index)
    neither = _make_adjacency_dataframe(empty.copy(), index)
    only1 = _make_adjacency_dataframe(empty.copy(), index)
    only2 = _make_adjacency_dataframe(empty.copy(), index)

    for i, source_group in enumerate(index):
        source_mask = group_indices == i
        for j, target_group in enumerate(index):
            target_mask = group_indices == j
            A1_subgraph = A1[source_mask][:, target_mask]
            A2_subgraph = A2[source_mask][:, target_mask]

            if i == j:
                rows, cols = offdiag_indices_from(A1_subgraph)
                edges1 = A1_subgraph[rows, cols]
                edges2 = A2_subgraph[rows, cols]
            else:
                edges1 = A1_subgraph.ravel()
                edges2 = A2_subgraph.ravel()

            curr_stat, curr_pvalue, curr_misc = binom_2samp_paired(edges1, edges2)

            stats.loc[source_group, target_group] = curr_stat
            uncorrected_pvalues.loc[source_group, target_group] = curr_pvalue
            both.loc[source_group, target_group] = curr_misc["n_both"]
            neither.loc[source_group, target_group] = curr_misc["n_neither"]
            only1.loc[source_group, target_group] = curr_misc["n_only1"]
            only2.loc[source_group, target_group] = curr_misc["n_only2"]

    misc = {}
    misc["both"] = both
    misc["neither"] = neither
    misc["only1"] = only1
    misc["only2"] = only2
    misc["uncorrected_pvalues"] = uncorrected_pvalues

    run_pvalues = uncorrected_pvalues.values
    run_pvalues = run_pvalues[~np.isnan(run_pvalues)]
    stat, pvalue = combine_pvalues(run_pvalues, method="fisher")
    n_tests = len(run_pvalues)
    misc["n_tests"] = n_tests
    return stat, pvalue, misc
