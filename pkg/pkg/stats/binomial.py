import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import test_proportions_2indep

from .fisher_exact_nonunity import fisher_exact_nonunity


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


def binom_2samp_paired(x, y):
    x = x.astype(bool)
    y = y.astype(bool)

    # TODO these two don't actually matter at all for McNemar's test...
    n_neither = ((~x) & (~y)).sum()
    n_both = (x & y).sum()

    n_only_x = (x & (~y)).sum()
    n_only_y = ((~x) & y).sum()

    cont_table = [[n_both, n_only_x], [n_only_y, n_neither]]
    cont_table = np.array(cont_table)

    bunch = mcnemar(cont_table)
    stat = bunch.statistic
    pvalue = bunch.pvalue

    return stat, pvalue
