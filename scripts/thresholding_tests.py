#%%
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.utils import remove_loops
from myst_nb import glue as default_glue
from pkg.data import DATA_VERSION, load_maggot_graph, select_nice_nodes
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import (
    compute_density_adjustment,
    erdos_renyi_test,
    stochastic_block_test,
)
from tqdm import tqdm

set_theme()

t0 = time.time()

DISPLAY_FIGS = True

FILENAME = "thresholding_tests"


def glue(name, var, prefix=None):
    savename = f"{FILENAME}-{name}"
    if prefix is not None:
        savename = prefix + ":" + savename
    default_glue(savename, var, display=False)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, prefix="fig")

    if not DISPLAY_FIGS:
        plt.close()


#%%

mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True)
left_nodes = left_mg.nodes
right_nodes = right_mg.nodes

left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj

GROUP_KEY = "simple_group"

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values
#%%


def binarize(A, threshold=None):
    # threshold is the smallest that is kept

    B = A.copy()

    if threshold is not None:
        B[B < threshold] = 0

    return B


rows = []
thresholds = np.arange(1, 10)
for threshold in tqdm(thresholds):
    left_adj_thresh = binarize(left_adj, threshold=threshold)
    right_adj_thresh = binarize(right_adj, threshold=threshold)

    stat, pvalue, misc = erdos_renyi_test(left_adj_thresh, right_adj_thresh)
    row = {"threshold": threshold, "stat": stat, "pvalue": pvalue, "method": "ER"}
    rows.append(row)

    for adjusted in [False, True]:
        if adjusted:
            adjustment_ratio = compute_density_adjustment(
                left_adj_thresh, right_adj_thresh
            )
            method = "SBM"
        else:
            adjustment_ratio = 1
            method = "aSBM"
        stat, pvalue, misc = stochastic_block_test(
            left_adj_thresh,
            right_adj_thresh,
            left_labels,
            right_labels,
            null_odds=adjustment_ratio,
        )
        row = {
            "threshold": threshold,
            "adjusted": adjusted,
            "stat": stat,
            "pvalue": pvalue,
            "method": method,
        }
        rows.append(row)
results = pd.DataFrame(rows)
results

#%%


fig, ax = plt.subplots(1, 1, figsize=(7, 6))

colors = sns.color_palette("tab20")
palette = dict(zip(["SBM", "aSBM", "ER"], colors))

sns.scatterplot(
    data=results, x="threshold", y="pvalue", hue="method", palette=palette, ax=ax
)
ax.set(yscale="log", ylabel="p-value", xlabel="Edge weight (# synapses) threshold")
ax.get_legend().set_title("Method")
ax.axhline(0.05, color="black", linestyle=":")
ax.text(ax.get_xlim()[1], 0.05, r"$\alpha = 0.05$", va="center", ha="left")
ax.set(xticks=thresholds)

gluefig("integer_threshold_pvalues", fig)

#%%

left_in_degrees = left_adj.sum(axis=0)
left_in_degrees[left_in_degrees == 0] = 1
left_adj_input_norm = left_adj / left_in_degrees[None, :]

right_in_degrees = right_adj.sum(axis=0)
right_in_degrees[right_in_degrees == 0] = 1
right_adj_input_norm = right_adj / right_in_degrees[None, :]


rows = []
thresholds = np.linspace(0, 0.03, 11)
for threshold in tqdm(thresholds):
    left_adj_thresh = binarize(left_adj_input_norm, threshold=threshold)
    right_adj_thresh = binarize(right_adj_input_norm, threshold=threshold)

    stat, pvalue, misc = erdos_renyi_test(left_adj_thresh, right_adj_thresh)
    row = {"threshold": threshold, "stat": stat, "pvalue": pvalue, "method": "ER"}
    rows.append(row)

    for adjusted in [False, True]:
        if adjusted:
            adjustment_ratio = compute_density_adjustment(
                left_adj_thresh, right_adj_thresh
            )
            method = "SBM"
        else:
            adjustment_ratio = 1
            method = "aSBM"
        stat, pvalue, misc = stochastic_block_test(
            left_adj_thresh,
            right_adj_thresh,
            left_labels,
            right_labels,
            null_odds=adjustment_ratio,
        )
        row = {
            "threshold": threshold,
            "adjusted": adjusted,
            "stat": stat,
            "pvalue": pvalue,
            "method": method,
        }
        rows.append(row)
results = pd.DataFrame(rows)
results

# %%


fig, ax = plt.subplots(1, 1, figsize=(7, 6))

colors = sns.color_palette("tab20")
palette = dict(zip(["SBM", "aSBM", "ER"], colors))

sns.scatterplot(
    data=results, x="threshold", y="pvalue", hue="method", palette=palette, ax=ax
)
ax.set(yscale="log", ylabel="p-value", xlabel="Edge weight (relative input) threshold")
ax.get_legend().set_title("Method")
ax.axhline(0.05, color="black", linestyle=":")
ax.text(ax.get_xlim()[1], 0.05, r"$\alpha = 0.05$", va="center", ha="left")
ax.set(xticks=thresholds)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))

gluefig("input_threshold_pvalues", fig)
