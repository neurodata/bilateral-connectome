import numpy as np
from scipy.stats import beta, chi2
from scipy.stats import combine_pvalues as scipy_combine_pvalues
from scipy.stats import ks_1samp, uniform


def combine_pvalues(pvalues, method="fisher"):
    pvalues = np.array(pvalues)

    scipy_methods = ["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]

    # scipy has a bug in these two methods
    if method == "pearson":  # HACK: https://github.com/scipy/scipy/pull/15452
        stat = 2 * np.sum(np.log1p(-pvalues))
        pvalue = chi2.cdf(-stat, 2 * len(pvalues))
    elif method == "tippett":  # HACK: https://github.com/scipy/scipy/pull/15452
        stat = np.min(pvalues)
        pvalue = beta.cdf(stat, 1, len(pvalues))
    elif method in scipy_methods:
        stat, pvalue = scipy_combine_pvalues(pvalues, method=method)
    elif method == "eric":  # provided for some experiments, not recommended
        stat, pvalue = ks_1samp(pvalues, uniform(0, 1).cdf, alternative="greater")
    elif method == "min":  # provided for some experiments, not recommended
        # very similar to Tippett's
        pvalue = min(pvalues.min() * len(pvalues), 1)
        stat = pvalue
    else:
        raise NotImplementedError()

    return stat, pvalue
