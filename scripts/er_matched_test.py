#%% [markdown]
# # A density-based test
# Here, we compare the two matched networks by treating each as an Erdos-Renyi network.

#%%

from giskard.plot.utils import soft_axis_off
from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import rotate_labels
from myst_nb import glue as default_glue
from pkg.data import load_matched, load_network_palette, load_node_palette
from pkg.io import savefig
from pkg.perturb import remove_edges
from pkg.plot import set_theme
from pkg.stats import erdos_renyi_test_paired

DISPLAY_FIGS = True
FILENAME = "er_matched_test"

rng = np.random.default_rng(8888)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue("fig:" + name, fig, display=False)

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var):
    default_glue(f"{FILENAME}-{name}", var, display=False)


t0 = time.time()
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()


left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")


#%%

stat, pvalue, misc = erdos_renyi_test_paired(left_adj, right_adj)
glue("pvalue", pvalue)

n_no_edge = misc["neither"]
n_both_edge = misc["both"]
n_only_left = misc["only1"]
n_only_right = misc["only2"]
glue("n_no_edge", n_no_edge)
glue("n_both_edge", n_both_edge)
glue("n_only_left", n_only_left)
glue("n_only_right", n_only_right)

#%%

# REF: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
fig, axs = plt.subplots(
    2,
    1,
    figsize=(8, 8),
    sharex=True,
    gridspec_kw=dict(hspace=0.05, height_ratios=[1, 2]),
)

neutral_color = sns.color_palette("Set2")[2]
empty_color = sns.color_palette("Set2")[7]


def plot_bar(x, height, color=None, ax=None, text=True):
    ax.bar(x, height, color=color)
    if text:
        ax.text(x, height, f"{height:,}", color=color, ha="center", va="bottom")


lower_ymax = max(n_both_edge, n_only_left, n_only_right)

ax = axs[0]
plot_bar(0, n_both_edge, color=neutral_color, ax=ax, text=False)
plot_bar(1, n_only_left, color=network_palette["Left"], ax=ax, text=False)
plot_bar(2, n_only_right, color=network_palette["Right"], ax=ax, text=False)
plot_bar(3, n_no_edge, color=empty_color, ax=ax)
ax.set_ylim(n_no_edge * 0.9, n_no_edge * 1.1)
ax.spines.bottom.set_visible(False)
ax.set_yticks([1_500_000])
ax.set_yticklabels(["1.5e6"])


ax = axs[1]
plot_bar(0, n_both_edge, color=neutral_color, ax=ax)
plot_bar(1, n_only_left, color=network_palette["Left"], ax=ax)
plot_bar(2, n_only_right, color=network_palette["Right"], ax=ax)
plot_bar(3, n_no_edge, color=empty_color, ax=ax, text=False)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["Edge in\nboth", "Left edge\nonly", "Right edge\nonly", "No edge"])
rotate_labels(ax)
ax.set_ylim(0, lower_ymax * 1.05)

ax.set_yticks([15000])
ax.set_yticklabels(["1.5e4"])

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
axs[1].plot([0, 1], [1, 1], transform=axs[1].transAxes, **kwargs)

gluefig("edge-count-bars", fig)

#%% [markdown]

# ```{glue:figure} fig:edge-count-bars
# :name: "fig:edge-count-bars"

# The number of edges in each of the four possible categories for the 2x2 paired
# contingency table comparing paired edges. P-value for the McNemar's test comparing
# the left and right paired edge counts is {glue:text}`er_matched_test-pvalue:0.3g`.
# Note that
# McNemar's test only compares the disagreeing edge counts, "Left edge only" and
# "Right edge only".
# ```

#%%
n_edges_left = np.count_nonzero(left_adj)
n_edges_right = np.count_nonzero(right_adj)
n_left = left_adj.shape[0]
n_right = right_adj.shape[0]
density_left = n_edges_left / (n_left ** 2)
density_right = n_edges_right / (n_right ** 2)

n_remove = int((density_right - density_left) * (n_right ** 2))

glue("density_left", density_left)
glue("density_right", density_right)
glue("n_remove", n_remove)

#%%
rows = []
n_resamples = 25
glue("n_resamples", n_resamples)
for i in range(n_resamples):
    subsampled_right_adj = remove_edges(
        right_adj, effect_size=n_remove, random_seed=rng
    )
    stat, pvalue, misc = erdos_renyi_test_paired(left_adj, subsampled_right_adj)
    rows.append({"stat": stat, "pvalue": pvalue, "misc": misc, "resample": i})

resample_results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(data=resample_results, x="pvalue", ax=ax)
if np.allclose(resample_results["pvalue"], 1):
    ax.axvline(resample_results.iloc[0]["pvalue"], color="darkred", linestyle="--")
ax.spines.left.set_visible(False)
ax.set(yticks=[], ylabel="", xlim=(-0.05, 1.05), xticks=[0, 0.5, 1], xlabel="p-value")
gluefig()

#%%
from pkg.stats import stochastic_block_test_paired

rows = []
n_resamples = 25
glue("n_resamples", n_resamples)
for i in range(n_resamples):
    subsampled_right_adj = remove_edges(
        right_adj, effect_size=n_remove, random_seed=rng
    )
    stat, pvalue, misc = stochastic_block_test_paired(
        left_adj, subsampled_right_adj, labels=left_nodes["simple_group"]
    )
    rows.append({"stat": stat, "pvalue": pvalue, "misc": misc, "resample": i})

resample_results = pd.DataFrame(rows)
resample_results

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
