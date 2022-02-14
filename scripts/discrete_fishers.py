#%%
from re import sub
from scipy.stats import binom, fisher_exact
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from giskard.plot import subuniformity_plot
from pkg.plot import set_theme
from scipy.stats import combine_pvalues

set_theme()
#%%
n = 10
p1 = 0.2
p2 = p1
n_sims = 100

pvalues = []
for i in range(n_sims):
    count1 = binom.rvs(n, p1)
    count2 = binom.rvs(n, p2)
    table = np.array([[count1, n - count1], [count2, n - count2]])
    stat, pvalue = fisher_exact(table)
    pvalues.append(pvalue)

subuniformity_plot(pvalues)

#%%
n = 1000
p1 = 0.007
p2 = 0.005
n_sims = 100

pvalues = []
for i in range(n_sims):
    # test 1
    count1 = binom.rvs(n, p1)
    count2 = binom.rvs(n, p1)
    table = np.array([[count1, n - count1], [count2, n - count2]])
    stat, pvalue1 = fisher_exact(table)

    count1 = binom.rvs(n, p2)
    count2 = binom.rvs(n, p2)
    table = np.array([[count1, n - count1], [count2, n - count2]])
    stat, pvalue2 = fisher_exact(table)

    stat, pvalue = combine_pvalues((pvalue1, pvalue2), method="fisher")

    pvalues.append(pvalue)

subuniformity_plot(pvalues)

#%%


def random_shift_pvalues(pvalues, rng=None):
    shifted_pvalues = np.array(pvalues).copy()  # redundant right?
    unique_pvalues, inverse = np.unique(pvalues, return_inverse=True)
    # pvalues = np.sort(pvalues)  # already makes a copy
    diffs = np.array(
        [unique_pvalues[0]] + list(unique_pvalues[1:] - unique_pvalues[:-1])
    )
    rng = np.random.default_rng()
    uniform_samples = rng.uniform(size=len(shifted_pvalues))
    matched_diffs = diffs[inverse]
    moves = uniform_samples * matched_diffs
    shifted_pvalues -= moves
    return shifted_pvalues


def discrete_combine_pvalues(pvalues, n_trials=100):
    pvalue = 0
    all_shifted_pvalues = []
    for i in range(n_trials):
        all_shifted_pvalues += list(random_shift_pvalues(pvalues))
    _, combined_pvalue = combine_pvalues(all_shifted_pvalues)
    # pvalue += combined_pvalue / n_trials  # TODO mean?
    return np.nan, combined_pvalue


def discrete_combine_pvalues(pvalues, n_trials=100):
    pvalue = 0
    for i in range(n_trials):
        shifted_pvalues = random_shift_pvalues(pvalues)
        _, combined_pvalue = combine_pvalues(shifted_pvalues)
        pvalue += combined_pvalue / n_trials  # TODO mean?
    return np.nan, combined_pvalue


#%%
import pandas as pd
from tqdm import tqdm

n_sims = 1000
n = 20
m = 10
rows = []
for i in tqdm(range(n_sims)):
    exact_test_pvalues = []
    for j in range(m):
        count1 = binom.rvs(n, 0.2)
        count2 = binom.rvs(n, 0.2)
        table = np.array([[count1, n - count1], [count2, n - count2]])
        stat, pvalue = fisher_exact(table)
        exact_test_pvalues.append(pvalue)

    _, pvalue = combine_pvalues(exact_test_pvalues)
    _, shifted_pvalue = discrete_combine_pvalues(exact_test_pvalues)
    rows.append({"pvalue": pvalue, "method": "fisher"})
    rows.append({"pvalue": shifted_pvalue, "method": "fisher-RS"})

results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(
    data=results,
    x="pvalue",
    hue="method",
    cumulative=True,
    element="step",
    ax=ax,
    stat="density",
    common_norm=False,
)
sns.move_legend(ax, "upper left", title="Method")
ax.set(ylabel="Cumulative density", xlabel="p-value")
ax.plot([0, 1], [0, 1], linestyle="--", color="black")

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# sns.histplot(x=pvalues, ax=ax, bins=200, stat="probability")
# ax.set(title="Null p-values - Fisher's exact test")

# subuniformity_plot(pvalues)
# #%%


# mod_pvalues = []
# for i in range(100):

#     # fisher_exact(shift)

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# sns.histplot(x=shifted_pvalues, ax=ax, bins=200, stat="probability")
# ax.set(title="Null p-values - Fisher's exact test")

# subuniformity_plot(shifted_pvalues)

#%%
# pvalues = [0.1, 0.2, 0.2, 0.8, 1]
# shifted_pvalues = np.array(pvalues)
# unique_pvalues, inverse = np.unique(pvalues, return_inverse=True)
# # pvalues = np.sort(pvalues)  # already makes a copy
# diffs = np.array([unique_pvalues[0]] + list(unique_pvalues[1:] - unique_pvalues[:-1]))
# rng = np.random.default_rng()
# uniform_samples = rng.uniform(size=len(pvalues))
# matched_diffs = diffs[inverse]
# moves = uniform_samples * matched_diffs
# shifted_pvalues -= moves

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# sns.histplot(x=shifted_pvalues, ax=ax, bins=200, stat="probability")
# ax.set(title="Null p-values - Fisher's exact test")

# # subuniformity_plot(shifted_pvalues)
