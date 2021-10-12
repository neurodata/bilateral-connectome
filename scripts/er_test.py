#%% [markdown]
# ## Preliminaries
#%%

import time

from pkg.data import (
    load_maggot_graph,
    load_network_palette,
    load_node_palette,
    select_nice_nodes,
)
from pkg.io import savefig
from pkg.plot import set_theme

from graspologic.utils import binarize
from scipy.stats import binom, norm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np


def stashfig(name, **kwargs):
    foldername = "er_test"
    savefig(name, foldername=foldername, **kwargs)


# %% [markdown]
# ## Load and process data
#%%

t0 = time.time()
set_theme()

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

#%%

mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True)


left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj
left_adj = binarize(left_adj)
right_adj = binarize(right_adj)

#%%

left_n = left_adj.size
right_n = right_adj.size
left_p = np.count_nonzero(left_adj) / left_n
right_p = np.count_nonzero(right_adj) / right_n

#%%

left_binom = binom(left_n, left_p)
mean = left_binom.mean()
x = np.arange(mean - 1000, mean + 1000)
pmf_vals = left_binom.pmf(x)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(x=x, y=pmf_vals, ax=ax)

mean = left_binom.mean()
std = left_binom.std()

left_norm = norm(mean, std)
pdf_vals = left_norm.pdf(x)
sns.lineplot(x=x, y=pdf_vals, ax=ax, linestyle="--")

#%%

ttest_ind(left_adj.ravel(), right_adj.ravel(), equal_var=False, alternative="two-sided")

#%%

from graspologic.simulations import er_np
from tqdm import tqdm

n_bootstraps = 500

pvalues = []
for i in tqdm(range(n_bootstraps)):
    A1 = er_np(len(left_adj), left_p, directed=True, loops=False)
    A2 = er_np(len(left_adj), left_p, directed=True, loops=False)
    stat, pvalue = ttest_ind(
        A1.ravel(), A2.ravel(), equal_var=False, alternative="two-sided"
    )
    pvalues.append(pvalue)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(pvalues, ax=ax, stat="density", cumulative=True)
from scipy.stats import ks_1samp, uniform

ks_1samp(pvalues, uniform(0, 1).cdf, alternative="greater")
