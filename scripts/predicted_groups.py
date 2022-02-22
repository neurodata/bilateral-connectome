#%%


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import rdpg_test
from pkg.utils import get_seeds

DISPLAY_FIGS = True
FILENAME = "predicted_groups"

rng = np.random.default_rng(8888)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, prefix="fig")

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var, prefix=None):
    savename = f"{FILENAME}-{name}"
    if prefix is not None:
        savename = prefix + ":" + savename
    default_glue(savename, var, display=False)


t0 = time.time()
set_theme()

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))

seeds = get_seeds(left_nodes, right_nodes)

#%%

n_components = 16
stat, pvalue, misc = rdpg_test(
    left_adj, right_adj, seeds=seeds, n_components=n_components
)
#%%
Z_left = misc["Z1"]
Z_right = misc["Z2"]

Z = np.concatenate((Z_left, Z_right), axis=0)

n_left = len(Z_left)
n_right = len(Z_right)

#%%

from scipy.cluster import hierarchy

linkages = hierarchy.linkage(Z, method="ward")

n_clusters = np.arange(5, 75, 10)
labels_by_cut = hierarchy.cut_tree(linkages, n_clusters=n_clusters)

#%%

from pkg.stats import stochastic_block_test

rows = []
for i, n_cluster in enumerate(n_clusters):
    all_labels = labels_by_cut[:, i]
    left_labels = all_labels[:n_left]
    right_labels = all_labels[n_left:]
    for density_adjustment in [True, False]:
        stat, pvalue, misc = stochastic_block_test(
            left_adj,
            right_adj,
            left_labels,
            right_labels,
            density_adjustment=density_adjustment,
            method="fisher",
            combine_method="tippett",
        )
        row = {
            "stat": stat,
            "pvalue": pvalue,
            "density_adjustment": density_adjustment,
            "i": i,
            "n_cluster": n_cluster,
        }
        rows.append(row)

results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.scatterplot(
    data=results, x="n_cluster", y="pvalue", hue="density_adjustment", ax=ax
)
ax.set_yscale("log")
