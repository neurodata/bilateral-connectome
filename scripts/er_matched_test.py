#%% [markdown]
# # A density-based test
# Here, we compare the two matched networks by treating each as an Erdos-Renyi network.

#%%

from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from giskard.plot import rotate_labels
from myst_nb import glue
from pkg.data import load_matched, load_network_palette, load_node_palette
from pkg.io import savefig
from pkg.plot import set_theme
from statsmodels.stats.contingency_tables import mcnemar

DISPLAY_FIGS = False
FILENAME = "er_matched_test"


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue("fig:" + name, fig, display=False)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()


left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")

#%%

edges1 = left_adj.ravel().astype(bool)
edges2 = right_adj.ravel().astype(bool)

n = left_adj.shape[0]
glue("n", n, display=False)

n_no_edge = ((~edges1) & (~edges2)).sum()
n_no_edge -= n  # ignore the diagonals
n_both_edge = (edges1 & edges2).sum()
n_only_1 = (edges1 & (~edges2)).sum()
n_only_2 = ((~edges1) & edges2).sum()
glue("n_no_edge", n_no_edge, display=False)
glue("n_both_edge", n_both_edge, display=False)
glue("n_only_left", n_only_1, display=False)
glue("n_only_right", n_only_2, display=False)

n_possible = n ** 2 - n
n_edge_sum = n_no_edge + n_both_edge + n_only_1 + n_only_2
assert n_possible == n_edge_sum
glue("n_possible", n_possible, display=False)

cont_table = [[n_both_edge, n_only_2], [n_only_1, n_no_edge]]
cont_table = np.array(cont_table)

bunch = mcnemar(cont_table)
stat = bunch.statistic
pvalue = bunch.pvalue
glue(f"{FILENAME}-pvalue", pvalue, display=False)

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


lower_ymax = max(n_both_edge, n_only_1, n_only_2)

ax = axs[0]
plot_bar(0, n_both_edge, color=neutral_color, ax=ax, text=False)
plot_bar(1, n_only_1, color=network_palette["Left"], ax=ax, text=False)
plot_bar(2, n_only_2, color=network_palette["Right"], ax=ax, text=False)
plot_bar(3, n_no_edge, color=empty_color, ax=ax)
ax.set_ylim(n_no_edge * 0.9, n_no_edge * 1.1)
ax.spines.bottom.set_visible(False)
ax.set_yticks([1_500_000])
ax.set_yticklabels(["1.5e6"])


ax = axs[1]
plot_bar(0, n_both_edge, color=neutral_color, ax=ax)
plot_bar(1, n_only_1, color=network_palette["Left"], ax=ax)
plot_bar(2, n_only_2, color=network_palette["Right"], ax=ax)
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
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
