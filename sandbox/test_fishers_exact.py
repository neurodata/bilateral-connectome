#%%
from scipy.stats import binom, fisher_exact
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pkg.plot import set_theme

set_theme()

n_trials = 10000

num = 3
denom = 18960

pvalues = []
for i in range(n_trials):
    x1 = binom(denom, num / denom).rvs()
    x2 = binom(denom, num / denom).rvs()
    table = np.array([[x1, denom - x1], [x2, denom - x2]])
    stat, pvalue = fisher_exact(table, alternative="two-sided")
    pvalues.append(pvalue)
pvalues = np.array(pvalues)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(x=pvalues, bins=np.linspace(0, 1, 11), stat="probability")
ax.tick_params(axis="x", length=5)

# %%
