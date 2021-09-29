from scipy.stats import ttest_ind


def erdos_reyni_test(A1, A2):
    # TODO should we replace with Fisher's exact test?
    stat, pvalue = ttest_ind(
        A1.ravel(), A2.ravel(), equal_var=False, alternative="two-sided"
    )
    return stat, pvalue, {}
