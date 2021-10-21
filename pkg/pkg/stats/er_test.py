from pkg.stats import stochastic_block_test
import numpy as np


# def erdos_renyi_test(A1, A2, method="fisher"):
#     triu_indices = np.triu_indices_from(A1, k=1)
#     tril_indices = np.tril_indices_from(A1, k=1)
#     A1_values = np.concatenate((A1[triu_indices], A1[tril_indices]))
#     A2_values = np.concatenate((A2[triu_indices], A2[tril_indices]))
#     m = len(A1_values)
#     if method == "t":
#         stat, pvalue = ttest_ind(
#             A1_values, A2_values, equal_var=False, alternative="two-sided"
#         )
#     elif method in ["chi2", "fisher"]:
#         A1_n_edges = np.count_nonzero(A1_values)
#         A2_n_edges = np.count_nonzero(A2_values)
#         A1_n_nonedges = m - A1_n_edges
#         A2_n_nonedges = m - A2_n_edges

#         # row is graph
#         # cols are (no edge, edge)
#         cont_table = np.array(
#             [
#                 [A1_n_nonedges, A1_n_edges],
#                 [A2_n_nonedges, A2_n_edges],
#             ]
#         )
#         if method == "chi2":
#             stat, pvalue, _, _ = chi2_contingency(cont_table)
#         elif method == "fisher":
#             stat, pvalue = fisher_exact(cont_table, alternative="two-sided")
#     return stat, pvalue, {}


def erdos_renyi_test(A1, A2, method="agresti-caffo"):
    stat, pvalue, misc = stochastic_block_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
    )
    return stat, pvalue, {}
