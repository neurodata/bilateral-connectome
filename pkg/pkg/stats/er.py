import numpy as np

from .sbm import stochastic_block_test, stochastic_block_test_paired


def _squeeze_value(old_misc, new_misc, old_key, new_key):
    variable = old_misc[old_key]
    variable = variable.values[0, 0]
    new_misc[new_key] = variable


def erdos_renyi_test(A1, A2, method="fisher"):
    stat, pvalue, sbm_misc = stochastic_block_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
    )
    old_keys = [
        "probabilities1",
        "probabilities2",
        "observed1",
        "observed2",
        "possible1",
        "possible2",
    ]
    new_keys = [
        "probability1",
        "probability2",
        "observed1",
        "observed2",
        "possible1",
        "possible2",
    ]
    er_misc = {}
    for old_key, new_key in zip(old_keys, new_keys):
        _squeeze_value(sbm_misc, er_misc, old_key, new_key)

    return stat, pvalue, er_misc


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
