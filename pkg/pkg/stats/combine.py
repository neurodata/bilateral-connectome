import numpy as np
from scipy.stats import chi2, beta
from scipy.stats import combine_pvalues as scipy_combine_pvalues


def combine_pvalues(pvalues, method="fisher"):
    pvalues = np.array(pvalues)
    # some methods use log(1 - pvalue) as part of the test statistic - thus when pvalue
    # is exactly 1 (which is possible for Fisher's exact test) we get an underfined
    # answer.
    # if pad_high > 0:
    #     upper_lim = 1 - pad_high
    #     pvalues[pvalues >= upper_lim] = upper_lim

    scipy_methods = ["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]

    if method == "fisher-discrete-random":
        # TODO was looking into methods for dealing with the discreteness of the
        # underlying tests
        raise NotImplementedError()
        # stat = 0
        # pvalue = 0
        # shifted_pvalues = []
        # for i in range(n_resamples):
        #     shifted_pvalues = random_shift_pvalues(pvalues)
        #     curr_stat, curr_pvalue = scipy_combine_pvalues(
        #         shifted_pvalues, method="fisher"
        #     )
        #     stat += curr_stat / n_resamples
        #     pvalue += curr_pvalue / n_resamples
    # scipy has a bug in these two methods
    elif method == "pearson":  # HACK: https://github.com/scipy/scipy/pull/15452
        stat = 2 * np.sum(np.log1p(-pvalues))
        pvalue = chi2.cdf(-stat, 2 * len(pvalues))
    elif method == "tippett":  # HACK: https://github.com/scipy/scipy/pull/15452
        stat = np.min(pvalues)
        pvalue = beta.cdf(stat, 1, len(pvalues))
    elif method in scipy_methods:
        stat, pvalue = scipy_combine_pvalues(pvalues, method=method)
    elif method == "eric":
        stat, pvalue = ks_1samp(pvalues, uniform(0, 1).cdf, alternative="greater")
    elif method == "min":
        pvalue = min(pvalues.min() * len(pvalues), 1)
        stat = pvalue
    else:
        raise NotImplementedError()

    return stat, pvalue
