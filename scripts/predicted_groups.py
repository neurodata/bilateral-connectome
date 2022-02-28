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
import pandas as pd

linkages = hierarchy.linkage(Z, method="ward")

n_clusters = np.arange(5, 75, 10)
labels_by_cut = hierarchy.cut_tree(linkages, n_clusters=n_clusters)

nodes = pd.concat((left_nodes, right_nodes))
for i in range(labels_by_cut.shape[1]):
    nodes[f"hier_labels_{i+1}"] = labels_by_cut[:, i]

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

#%%

from giskard.plot import crosstabplot

group = "hier_labels_5"
ax = crosstabplot(data=nodes, group=group, hue="simple_group", palette=node_palette)

ax.set_title(group)
K = nodes[group].nunique()
ax.text(0.8, 0.8, f"K = {K}", transform=ax.transAxes)

#%%


rows = []
for i in range(1, 9):
    # all_labels = labels_by_cut[:, i]
    # left_labels = all_labels[:n_left]
    # right_labels = all_labels[n_left:]
    left_labels = left_nodes[
        f"dc_level_{i}_n_components=10_min_split=32"
    ].values.astype(int)
    right_labels = right_nodes[
        f"dc_level_{i}_n_components=10_min_split=32"
    ].values.astype(int)
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
        n_cluster = misc["uncorrected_pvalues"].shape[0]
        row = {
            "stat": stat,
            "pvalue": pvalue,
            "density_adjustment": density_adjustment,
            "i": i,
            "n_cluster": n_cluster,
            "level": i,
        }
        rows.append(row)

results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.scatterplot(
    data=results, x="n_cluster", y="pvalue", hue="density_adjustment", ax=ax
)
sns.lineplot(
    data=results,
    x="n_cluster",
    y="pvalue",
    hue="density_adjustment",
    ax=ax,
    legend=False,
)
ax.get_legend().set_title("Density adjustment")
ax.set_yscale("log")
ax.set(ylabel="p-value", xlabel="# of neuron groups")
ax.axhline(0.05, color="black", linestyle=":")
ax.text(ax.get_xlim()[1], 0.05, r"$\alpha = 0.05$", va="center", ha="left")

gluefig("dendrogram_clusters_pvalues", fig)

#%%
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
