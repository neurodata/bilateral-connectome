#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.plot import set_theme
from scipy.stats import binom, chi2_contingency, ks_1samp, uniform

c = 1.5
ns = np.array([1000, 1500, 750, 1250])
ps = np.array([0.2, 0.02, 0.05, 0.1])

rng = np.random.default_rng(8888)


# for i in range(len(ns)):

K = len(ns)

n_sims = 1000
rows = []
for sim in range(n_sims):
    for i in range(K):
        x = rng.binomial(ns, ps)
        y = rng.binomial(ns, c * ps)

        cont_table = np.array([[x[i], ns[i] - x[i]], [y[i], ns[i] - y[i]]])
        stat, pvalue, _, _ = chi2_contingency(cont_table)
        rows.append(
            {
                "dim": i,
                "stat": stat,
                "pvalue": pvalue,
                "sim": sim,
                "c": c,
                "correction": False,
            }
        )

        corrected_cont_table = cont_table.copy()
        # corrected_cont_table[1, 0] = cont_table[1, 0] +
        n = corrected_cont_table[1].sum()
        phat = corrected_cont_table[1, 0] / n
        phat = 1 / c * phat
        corrected_cont_table[1, 0] = phat * n
        corrected_cont_table[1, 1] = (1 - phat) * n
        stat, pvalue, _, _ = chi2_contingency(corrected_cont_table)
        rows.append(
            {
                "dim": i,
                "stat": stat,
                "pvalue": pvalue,
                "sim": sim,
                "c": c,
                "correction": True,
            }
        )

results = pd.DataFrame(rows)

#%%

set_theme()
for i in range(K):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(data=results[results["dim"] == i], x="pvalue", hue="correction", ax=ax)


def subuniformity_plot(x, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(x, ax=ax, stat="density", cumulative=True, **kwargs)
    stat, pvalue = ks_1samp(x, uniform(0, 1).cdf, alternative="greater")
    ax.plot([0, 1], [0, 1], linewidth=3, linestyle=":", color="black")
    ax.text(0, 1, f"p-value: {pvalue:.3f}")
    ax.set_ylabel("Cumulative density")
    return ax, stat, pvalue


for i in range(K):
    data = results[(results["dim"] == i) & (results["correction"])]["pvalue"]
    subuniformity_plot(data)

#%%


ps = np.linspace(0.1, 0.3, 1000)

x = 30  # p ~= 0.2
n = 100

y = 40  # p ~= 0.3 so c ~= 1.5
m = 200

c = 1.5


def func(p):
    return (
        x * np.log(c * p)
        + (n - x) * np.log(1 - c * p)
        + y * np.log(p)
        + (m - y) * np.log(1 - p)
    )


vals = func(ps)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(ps, vals)

est = 1 / c * x / (n + m) + y / (n + m)

ax.axvline(est, color="red")
