#%% [markdown]
# ## Preliminaries
#%%

import time
from giskard.plot import remove_shared_ax

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_random_seed
from graspologic.utils import binarize
from pkg.data import (
    load_maggot_graph,
    load_network_palette,
    load_node_palette,
    select_nice_nodes,
)
from pkg.io import savefig
from pkg.perturb import add_edges, remove_edges, shuffle_edges
from pkg.plot import set_theme
from pkg.stats import stochastic_block_test
from seaborn.utils import relative_luminance


def stashfig(name, **kwargs):
    foldername = "sbm_test"
    savefig(name, foldername=foldername, **kwargs)


# %% [markdown]
# ## Load and process data
#%%

t0 = time.time()
set_theme()

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

#%%

GROUP_KEY = "simple_group"

mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True)
left_nodes = left_mg.nodes
right_nodes = right_mg.nodes

left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj
left_adj = binarize(left_adj)
right_adj = binarize(right_adj)

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values


from pkg.stats import fit_sbm

B1, n_observed1, n_possible1 = fit_sbm(left_adj, left_labels)

B1

#%%

from giskard.plot import remove_shared_ax


def rotate_labels(ax):
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


stat, pvalue, misc = stochastic_block_test(
    left_adj, right_adj, labels1=left_labels, labels2=right_labels, method="fisher"
)

B1 = misc["probabilities1"]
B2 = misc["probabilities2"]
index = B1.index
p_max = max(B1.values.max(), B2.values.max())
uncorrected_pvalues = misc["uncorrected_pvalues"]

alpha = 0.05
K = B1.shape[0]
hb_thresh = alpha / K ** 2

set_theme(font_scale=1.25)
fig, axs = plt.subplots(
    2,
    5,
    figsize=(30, 15),
    gridspec_kw=dict(
        height_ratios=[2, 1], width_ratios=[0.3, 10, 10, 10, 1], hspace=0.1
    ),
    constrained_layout=True,
)
axs[1, 1].sharex(axs[0, 1])
heatmap_kws = dict(cmap="Blues", square=True, cbar=False, vmax=p_max)

ax = axs[0, 1]
sns.heatmap(B1, ax=ax, **heatmap_kws)
ax.set(ylabel="Source group", xlabel="Target group")
ax.set_title(r"$\hat{B}$ left", fontsize="xx-large")
# rotate_labels(ax)
# plt.setp(ax.get_xticklabels(), visible=True)


ax = axs[0, 2]
im = sns.heatmap(B2, ax=ax, **heatmap_kws)
ax.set(ylabel="", xlabel="Target group")
ax.set_title(r"$\hat{B}$ right", fontsize="xx-large")
# rotate_labels(ax)

ax = axs[0, 0]
cbar = fig.colorbar(
    im.get_children()[0],
    ax=ax,
    fraction=1,
    shrink=1,
    ticklocation="left",
    location="left",
)
ax.axis("off")
# ax.remove()

ax = axs[0, 3]

colors = im.get_children()[0].get_facecolors()
mask = uncorrected_pvalues < hb_thresh

# annot = pd.DataFrame(np.full((K, K), "*"), index=mask.index, columns=mask.columns)
# annot[~mask] = ""
annot = False

heatmap_kws = dict(cmap="RdBu", center=0, square=True, cbar=False)
im = sns.heatmap(
    np.log10(uncorrected_pvalues), ax=ax, annot=annot, fmt="", **heatmap_kws
)
ax.set(ylabel="", xlabel="Target group")
ax.set_title(r"$log($p-value$)$ (unadjusted)", fontsize="xx-large")


pad = 0.2
for idx, (is_significant, color) in enumerate(zip(mask.values.ravel(), colors)):
    if is_significant:
        i, j = np.unravel_index(idx, (K, K))
        lum = relative_luminance(color)
        text_color = ".15" if lum > 0.408 else "w"

        xs = [j + pad, j + 1 - pad]
        ys = [i + pad, i + 1 - pad]
        ax.plot(xs, ys, color=text_color, linewidth=4)
        xs = [j + 1 - pad, j + pad]
        ys = [i + pad, i + 1 - pad]
        ax.plot(xs, ys, color=text_color, linewidth=4)


ax = axs[0, 4]
fig.colorbar(
    im.get_children()[0],
    ax=ax,
    fraction=0.6,
    shrink=0.6,
)
ax.remove()


def countplot(group_counts, ax):
    for i in range(len(group_counts)):
        ax.bar(i + 0.5, group_counts[i], color="dimgray")
    ax.set(ylabel="Count", xlabel="Group", xticks=[])
    # ax.autoscale()


ax = axs[1, 1]
# ax.sharex(axs[0, 1])
countplot(misc["group_counts1"], ax)
ax.set_title("Left", fontsize="xx-large")
remove_shared_ax(ax, y=False)
axs[0, 1].set_xticks(np.arange(K) + 0.5)
axs[0, 1].set_xticklabels(index)

ax = axs[1, 2]
ax.sharex(axs[0, 2])
countplot(misc["group_counts2"], ax)
ax.set_title("Right", fontsize="xx-large")
remove_shared_ax(ax, y=False)
axs[0, 2].set_xticks(np.arange(K) + 0.5)
axs[0, 2].set_xticklabels(index)

ax = axs[1, 3]
ax.axis("off")
ax.text(0, 0.7, f"Overall p-value: {pvalue:0.2e}", fontsize="x-large", ha="left")
ax.text(
    0,
    0.3,
    r"$\times$ denotes significant after"
    + "\nBonferonni-Holm correction,"
    + r" $\alpha=0.05$",
    fontsize="x-large",
)
# i = 0.5
# j = 0.5
# xs = [j, j + 0.01]
# ys = [i, i + 0.01]
# ax.plot(xs, ys, color=text_color, linewidth=4)
# xs = [j + 0.01, j]
# ys = [i, i + 0.01]
# ax.plot(xs, ys, color=text_color, linewidth=4)

axs[1, 0].remove()
axs[1, 4].remove()

stashfig("SBM-left-right-comparison")
# %%
