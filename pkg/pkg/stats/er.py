import numpy as np

from .sbm_test import stochastic_block_test


def erdos_renyi_test(A1, A2, method="fisher"):
    stat, pvalue, _ = stochastic_block_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
        null_odds=1,
    )
    return stat, pvalue, _
