#%% [markdown]
# (page:nufe)=
# # An exact test for non-unity null odds ratios
#%%
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import set_theme, subuniformity_plot
from matplotlib.figure import Figure
from myst_nb import glue
from scipy.stats import binom, fisher_exact, nchypergeom_fisher

set_theme()


def my_glue(name, variable):
    glue(name, variable, display=False)
    if isinstance(variable, Figure):
        plt.close()


#%% [markdown]
# ## Code for the test
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


#%% [markdown]
# ## Simulation setup
# Here, we investigate the performance of this test on simulated independent binomials.
# The model we use is as follows:
#
# Let
#
# $$X \sim Binomial(m_x, p_x)$$
#
# independently,
#
# $$Y \sim Binomial(m_y, p_y)$$
#
# Fix $m_x = $ {glue:text}`m_x`, $m_y = $ {glue:text}`m_y`, $p_y = $ {glue:text}`p_y`.
#
# Let $p_x = \omega p_y$, where $\omega$ is some positive real number, not
# not necessarily equal to 1, such that $p_x \in (0, 1)$.
#%%

m_x = 1000
my_glue("m_x", m_x)
m_y = 2000
my_glue("m_y", m_y)

upper_omegas = np.linspace(1, 4, 7)
lower_omegas = 1 / upper_omegas
lower_omegas = lower_omegas[1:]
omegas = np.sort(np.concatenate((upper_omegas, lower_omegas)))
my_glue("upper_omega", max(omegas))
my_glue("lower_omega", min(omegas))


p_y = 0.01
my_glue("p_y", p_y)

n_sims = 200
my_glue("n_sims", n_sims)

alternative = "two-sided"
#%% [markdown]
# ## Experiment
# Below, we'll sample from the model described above, varying $\omega$ from
# {glue:text}`lower_omega` to {glue:text}`upper_omega`. For each value of $\omega$,
# we'll draw {glue:text}`n_sims` samples of $(x,y)$ from the model described above.
# For each draw, we'll test the following two hypotheses:
#
# $$H_0: p_x = p_y \quad H_A: p_x \neq p_y$$
# using Fisher's exact test, and
#
# $$H_0: p_x = \omega_0 p_y \quad H_A: p_x \neq \omega_0 p_y$$
# using a modified version of Fisher's exact test, which uses
# [Fisher's noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution)
# as the null distribution. Note that we can re-write this null as
#
# $$H_0: \frac{p_x}{p_y} = \omega_0$$
#
# to easily see that this is a test for a posited odds ratio $\omega_0$. For this
# experiment, we set $\omega_0 = \omega$ - in other words, we assume the true odds ratio
# is known.
#
# Below, we'll call the first hypothesis test **FE** (Fisher's Exact), and the second
# **NUFE (Non-unity Fisher's Exact)**.

#%%

# definitions following https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution

rows = []
for omega in omegas:
    # params
    p_x = omega * p_y
    omega_x = p_x / (1 - p_x)
    omega_y = p_y / (1 - p_y)

    for sim in range(n_sims):
        # sample
        x = binom.rvs(m_x, p_x)
        y = binom.rvs(m_y, p_y)
        n = x + y

        table = np.array([[x, m_x - x], [y, m_y - y]])
        _, vanilla_pvalue = fisher_exact(table, alternative=alternative)
        _, nu_pvalue = fisher_exact_nonunity(
            table, alternative=alternative, null_odds=omega
        )

        rows.append(
            {
                "method": "FE",
                "omega": omega,
                "pvalue": vanilla_pvalue,
                "sim": sim,
            }
        )

        rows.append({"method": "NUFE", "omega": omega, "pvalue": nu_pvalue, "sim": sim})


results = pd.DataFrame(rows)

#%% [markdown]
# ## Results
#%% [markdown]
#%%

colors = sns.color_palette()
palette = {"FE": colors[0], "NUFE": colors[1]}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="omega", y="pvalue", hue="method", ax=ax, palette=palette)
ax.axvline(1, color="darkred", linestyle="--")
ax.set(ylabel="p-value", xlabel=r"$\omega$")

my_glue("fig_mean_pvalue_by_omega", fig)


#%% [markdown]
# ```{glue:figure} fig_mean_pvalue_by_omega
# :name: "fig-mean-pvalue-by-omega"
#
# Plot of mean p-values by the true odds ratio $\omega$. Each point in the lineplot
# denotes the mean over {glue:text}`n_sims` trials, and the shaded region denotes 95%
# bootstrap CI estimates. Fisher's Exact test
# (FE) tends to reject for non-unity odds ratios, while the modified, non-unity Fisher's
# exact test (NUFE) does not. Note that NUFE is testing the null hypothesis that the
# odds ratio ($\omega_0$) is equal to the true odds ratio in the simulation ($\omega$).
# ```

#%%


def plot_select_pvalues(method, omega, ax):

    subuniformity_plot(
        results[(results["method"] == method) & (results["omega"] == omega)][
            "pvalue"
        ].values,
        color=palette[method],
        ax=ax,
    )
    ax.set(title=r"$\omega = $" + f"{omega}, method = {method}", xlabel="p-value")


fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_select_pvalues(method="FE", omega=1, ax=axs[0])
plot_select_pvalues(method="NUFE", omega=1, ax=axs[1])

my_glue("fig_pvalue_dist_omega_1", fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_select_pvalues(method="FE", omega=3, ax=axs[0])
plot_select_pvalues(method="NUFE", omega=3, ax=axs[1])

my_glue("fig_pvalue_dist_omega_3", fig)

#%% [markdown]
# ```{glue:figure} fig_pvalue_dist_omega_1
# :name: "fig-pvalue-dist-omega-1"
#
# Cumulative distribution of p-values for both tests when the true odds ratio
# $\omega = 1$. Dashed line denotes the cumulative distribution function for a
# $Uniform(0,1)$ random variable. p-value in upper left is for a one-sided KS-test that
# for the null that the distribution is stochastically smaller than a $Uniform(0,1)$
# random variable. Note that both tests appear to have uniform p-values in the case
# where the null that both tests are concerned with is true.
# ```

#%% [markdown]
# ```{glue:figure} fig_pvalue_dist_omega_3
# :name: "fig-pvalue-dist-omega-3"
#
# Cumulative distribution of p-values for both tests when the true odds ratio
# $\omega = 3$. Dashed line denotes the cumulative distribution function for a
# $Uniform(0,1)$ random variable. p-value in upper left is for a one-sided KS-test that
# for the null that the distribution is stochastically smaller than a $Uniform(0,1)$
# random variable. When $\omega = 3$, the null that FE is testing is indeed false, so it
# makes sense that we see yield many small p-values. However, for $NUFE$, $H_0$ is true,
# as we specified that $\omega_0 = \omega$. NUFE remains valid in this case as the
# p-values appear stochastically smaller than a $Uniform(0,1)$ random variable.
# ```
