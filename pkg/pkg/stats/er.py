import numpy as np

from .sbm import stochastic_block_test, stochastic_block_test_paired


def erdos_renyi_test(A1, A2, method="fisher"):
    stat, pvalue, _ = stochastic_block_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
        null_odds=1,
    )
    return stat, pvalue, {}


def erdos_renyi_test_paired(A1, A2):
    stat, pvalue, sbm_misc = stochastic_block_test_paired(
        A1, A2, labels=np.ones(A1.shape[0])
    )
    misc = {}
    misc["both"] = sbm_misc["both"].loc[1, 1]
    misc["neither"] = sbm_misc["neither"].loc[1, 1]
    misc["only1"] = sbm_misc["only1"].loc[1, 1]
    misc["only2"] = sbm_misc["only2"].loc[1, 1]
    return stat, pvalue, misc
