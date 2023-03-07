#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import set_theme
from graspologic.simulations import sbm
from pkg.stats import stochastic_block_test
from scipy.stats import ks_1samp, uniform

set_theme()

B = np.array([[0.4, 0.1], [0.1, 0.3]])
Delta = np.array([[-0.1, 0.05], [-0.05, 0.1]])
alpha = 0
n = 100
n_per_comm = [25, 75]


n_trials = 200
rows = []
for i in range(n_trials):
    A1, labels1 = sbm(
        n_per_comm, B + alpha * Delta, loops=False, directed=True, return_labels=True
    )
    A2, labels2 = sbm(n_per_comm, B, loops=False, directed=True, return_labels=True)

    stat, pvalue, _ = stochastic_block_test(A1, A2, labels1, labels2, method="chi2")
    rows.append({"stat": stat, "pvalue": pvalue})

results = pd.DataFrame(rows)

results

#%%


def subuniformity_plot(x, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(x, ax=ax, stat="density", cumulative=True, **kwargs)
    stat, pvalue = ks_1samp(x, uniform(0, 1).cdf, alternative="greater")
    ax.plot([0, 1], [0, 1], linewidth=3, linestyle=":", color="black")
    ax.text(0, 1, f"p-value: {pvalue:.3f}")
    ax.set_ylabel("Cumulative density")
    return ax, stat, pvalue


subuniformity_plot(results["pvalue"])
