#%%
from scipy.stats import nchypergeom_fisher, binom, fisher_exact
import matplotlib.pyplot as plt
import numpy as np
from giskard.plot import set_theme
from scipy.stats import hypergeom

# definitions following https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution
m_x = 1000
m_y = 2000
omega = 3
pi_y = 0.01
pi_x = omega * pi_y
omega_x = pi_x / (1 - pi_x)
omega_y = pi_y / (1 - pi_y)

x = binom.rvs(m_x, pi_x)
y = binom.rvs(m_y, pi_y)
n = x + y


omega_hat = (x / m_x) * (1 - y / m_y) / ((y / m_y) * (1 - x / m_x))

omega_naught = 1

rv = nchypergeom_fisher(m_y + m_x, m_x, n, omega_naught)

# set_theme()

# xs = np.arange(rv.ppf(0.01), rv.ppf(0.99))
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# ax.vlines(xs, 0, rv.pmf(xs), colors="k", linestyles="-", lw=2)

# lower = rv.ppf(0.01)
# upper = rv.ppf(0.99)
# print(lower)
# print(upper)
# print(x)

print("noncentral centralized")
# probability (T \leq t) observed under the null
print(rv.cdf(x))
# probability (T \geq t) observed under the null
print(rv.sf(x - 1))


print("hypergeom")

rv = hypergeom(m_y + m_x, m_x, n)
print(rv.cdf(x))
print(rv.sf(x - 1))

print("fisher scipy")
cont_table = np.array([[x, m_x - x], [y, m_y - y]])
stat, pvalue = fisher_exact(cont_table, alternative="less")
print(pvalue)

#%%


def fisher_exact_nonunity(table, alternative="two-sided", null_odds=1):
    """Perform a Fisher exact test on a 2x2 contingency table.
    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
        * 'two-sided'
        * 'less': one-sided
        * 'greater': one-sided
        See the Notes for more details.
    Returns
    -------
    oddsratio : float
        This is prior odds ratio and not a posterior estimate.
    p_value : float
        P-value, the probability of obtaining a distribution at least as
        extreme as the one that was actually observed, assuming that the
        null hypothesis is true.
    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.  This can be used as an alternative to
        `fisher_exact` when the numbers in the table are large.
    barnard_exact : Barnard's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.
    boschloo_exact : Boschloo's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.
    Notes
    -----
    *Null hypothesis and p-values*
    The null hypothesis is that the input table is from the hypergeometric
    distribution with parameters (as used in `hypergeom`)
    ``M = a + b + c + d``, ``n = a + b`` and ``N = a + c``, where the
    input table is ``[[a, b], [c, d]]``.  This distribution has support
    ``max(0, N + n - M) <= x <= min(N, n)``, or, in terms of the values
    in the input table, ``min(0, a - d) <= x <= a + min(b, c)``.  ``x``
    can be interpreted as the upper-left element of a 2x2 table, so the
    tables in the distribution have form::
        [  x           n - x     ]
        [N - x    M - (n + N) + x]
    For example, if::
        table = [6  2]
                [1  4]
    then the support is ``2 <= x <= 7``, and the tables in the distribution
    are::
        [2 6]   [3 5]   [4 4]   [5 3]   [6 2]  [7 1]
        [5 0]   [4 1]   [3 2]   [2 3]   [1 4]  [0 5]
    The probability of each table is given by the hypergeometric distribution
    ``hypergeom.pmf(x, M, n, N)``.  For this example, these are (rounded to
    three significant digits)::
        x       2      3      4      5       6        7
        p  0.0163  0.163  0.408  0.326  0.0816  0.00466
    These can be computed with::
        >>> from scipy.stats import hypergeom
        >>> table = np.array([[6, 2], [1, 4]])
        >>> M = table.sum()
        >>> n = table[0].sum()
        >>> N = table[:, 0].sum()
        >>> start, end = hypergeom.support(M, n, N)
        >>> hypergeom.pmf(np.arange(start, end+1), M, n, N)
        array([0.01631702, 0.16317016, 0.40792541, 0.32634033, 0.08158508,
               0.004662  ])
    The two-sided p-value is the probability that, under the null hypothesis,
    a random table would have a probability equal to or less than the
    probability of the input table.  For our example, the probability of
    the input table (where ``x = 6``) is 0.0816.  The x values where the
    probability does not exceed this are 2, 6 and 7, so the two-sided p-value
    is ``0.0163 + 0.0816 + 0.00466 ~= 0.10256``::
        >>> from scipy.stats import fisher_exact
        >>> oddsr, p = fisher_exact(table, alternative='two-sided')
        >>> p
        0.10256410256410257
    The one-sided p-value for ``alternative='greater'`` is the probability
    that a random table has ``x >= a``, which in our example is ``x >= 6``,
    or ``0.0816 + 0.00466 ~= 0.08626``::
        >>> oddsr, p = fisher_exact(table, alternative='greater')
        >>> p
        0.08624708624708627
    This is equivalent to computing the survival function of the
    distribution at ``x = 5`` (one less than ``x`` from the input table,
    because we want to include the probability of ``x = 6`` in the sum)::
        >>> hypergeom.sf(5, M, n, N)
        0.08624708624708627
    For ``alternative='less'``, the one-sided p-value is the probability
    that a random table has ``x <= a``, (i.e. ``x <= 6`` in our example),
    or ``0.0163 + 0.163 + 0.408 + 0.326 + 0.0816 ~= 0.9949``::
        >>> oddsr, p = fisher_exact(table, alternative='less')
        >>> p
        0.9953379953379957
    This is equivalent to computing the cumulative distribution function
    of the distribution at ``x = 6``:
        >>> hypergeom.cdf(6, M, n, N)
        0.9953379953379957
    *Odds ratio*
    The calculated odds ratio is different from the one R uses. This SciPy
    implementation returns the (more common) "unconditional Maximum
    Likelihood Estimate", while R uses the "conditional Maximum Likelihood
    Estimate".
    Examples
    --------
    Say we spend a few days counting whales and sharks in the Atlantic and
    Indian oceans. In the Atlantic ocean we find 8 whales and 1 shark, in the
    Indian ocean 2 whales and 5 sharks. Then our contingency table is::
                Atlantic  Indian
        whales     8        2
        sharks     1        5
    We use this table to find the p-value:
    >>> from scipy.stats import fisher_exact
    >>> oddsratio, pvalue = fisher_exact([[8, 2], [1, 5]])
    >>> pvalue
    0.0349...
    The probability that we would observe this or an even more imbalanced ratio
    by chance is about 3.5%.  A commonly used significance level is 5%--if we
    adopt that, we can therefore conclude that our observed imbalance is
    statistically significant; whales prefer the Atlantic while sharks prefer
    the Indian ocean.
    """
    dist = nchypergeom_fisher

    # int32 is not enough for the algorithm
    c = np.asarray(table, dtype=np.int64)
    if not c.shape == (2, 2):
        raise ValueError("The input `table` must be of shape (2, 2).")

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is 1 and
        # the odds ratio is NaN.
        return np.nan, 1.0

    if c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf

    n1 = c[0, 0] + c[0, 1]
    n2 = c[1, 0] + c[1, 1]
    n = c[0, 0] + c[1, 0]

    rv = dist(n1 + n2, n1, n, null_odds)

    def binary_search(n, n1, n2, side):
        """Binary search for where to begin halves in two-sided test."""
        if side == "upper":
            minval = mode
            maxval = n
        else:
            minval = 0
            maxval = mode
        guess = -1
        while maxval - minval > 1:
            if maxval == minval + 1 and guess == minval:
                guess = maxval
            else:
                guess = (maxval + minval) // 2
            pguess = rv.pmf(guess)
            if side == "upper":
                ng = guess - 1
            else:
                ng = guess + 1
            if pguess <= pexact < rv.pmf(ng):
                break
            elif pguess < pexact:
                maxval = guess
            else:
                minval = guess
        if guess == -1:
            guess = minval
        if side == "upper":
            while guess > 0 and rv.pmf(guess) < pexact * epsilon:
                guess -= 1
            while rv.pmf(guess) > pexact / epsilon:
                guess += 1
        else:
            while rv.pmf(guess) < pexact * epsilon:
                guess += 1
            while guess > 0 and rv.pmf(guess) > pexact / epsilon:
                guess -= 1
        return guess

    if alternative == "less":
        pvalue = rv.cdf(c[0, 0])
    elif alternative == "greater":
        # Same formula as the 'less' case, but with the second column.
        pvalue = rv.sf(c[0, 0] - 1)
    elif alternative == "two-sided":
        mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
        pexact = dist.pmf(c[0, 0], n1 + n2, n1, n, null_odds)
        pmode = dist.pmf(mode, n1 + n2, n1, n, null_odds)

        epsilon = 1 - 1e-4
        if np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= 1 - epsilon:
            return oddsratio, 1.0

        elif c[0, 0] < mode:
            plower = dist.cdf(c[0, 0], n1 + n2, n1, n, null_odds)
            if dist.pmf(n, n1 + n2, n1, n, null_odds) > pexact / epsilon:
                return oddsratio, plower

            guess = binary_search(n, n1, n2, "upper")
            pvalue = plower + dist.sf(guess - 1, n1 + n2, n1, n, null_odds)
        else:
            pupper = dist.sf(c[0, 0] - 1, n1 + n2, n1, n, null_odds)
            if dist.pmf(0, n1 + n2, n1, n, null_odds) > pexact / epsilon:
                return oddsratio, pupper

            guess = binary_search(n, n1, n2, "lower")
            pvalue = pupper + dist.cdf(guess, n1 + n2, n1, n, null_odds)
    else:
        msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}"
        raise ValueError(msg)

    pvalue = min(pvalue, 1.0)

    return oddsratio, pvalue


#%%
import pandas as pd
from tqdm import tqdm

m_x = 1000
m_y = 2000

alternative = "two-sided"
omegas = np.linspace(0.25, 4, 20)
pi_y = 0.01
n_sims = 200
rows = []
for omega in omegas:
    # params
    pi_x = omega * pi_y
    omega_x = pi_x / (1 - pi_x)
    omega_y = pi_y / (1 - pi_y)

    for sim in tqdm(range(n_sims)):
        # sample
        x = binom.rvs(m_x, pi_x)
        y = binom.rvs(m_y, pi_y)
        n = x + y

        table = np.array([[x, m_x - x], [y, m_y - y]])
        _, nu_pvalue = fisher_exact_nonunity(
            table, alternative=alternative, null_odds=omega
        )
        _, vanilla_pvalue = fisher_exact(table, alternative=alternative)

        rows.append(
            {"method": "FNCHG", "omega": omega, "pvalue": nu_pvalue, "sim": sim}
        )
        rows.append(
            {"method": "HG", "omega": omega, "pvalue": vanilla_pvalue, "sim": sim}
        )

results = pd.DataFrame(rows)
#%%
results

#%%
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="omega", y="pvalue", hue="method", ax=ax)
ax.axvline(1, color="darkred", linestyle="--")
ax.set(ylabel="p-value", xlabel=r"$\omega$")
