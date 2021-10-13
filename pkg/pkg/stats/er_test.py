from scipy.stats import ttest_ind, fisher_exact, chi2_contingency
import numpy as np


def erdos_reyni_test(A1, A2, method="fisher"):
    triu_indices = np.triu_indices_from(A1, k=1)
    tril_indices = np.tril_indices_from(A1, k=1)
    A1_values = np.concatenate((A1[triu_indices], A1[tril_indices]))
    A2_values = np.concatenate((A2[triu_indices], A2[tril_indices]))
    m = len(A1_values)
    if method == "t":
        stat, pvalue = ttest_ind(
            A1_values, A2_values, equal_var=False, alternative="two-sided"
        )
    elif method == "chisq":
        raise NotImplementedError()
    elif method == "fisher":
        A1_n_edges = np.count_nonzero(A1_values)
        A2_n_edges = np.count_nonzero(A2_values)
        A1_n_nonedges = m - A1_n_edges
        A2_n_nonedges = m - A2_n_edges

        # row is graph
        # cols are (no edge, edge)
        cont_table = np.array(
            [
                [A1_n_nonedges, A1_n_edges],
                [A2_n_nonedges, A2_n_edges],
            ]
        )
        stat, pvalue = fisher_exact(cont_table, alternative="two-sided")
    return stat, pvalue, {}
