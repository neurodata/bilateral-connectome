#%% [markdown]
# # A group-based test
# Next, we test bilateral symmetry by making an assumption that the left and the right
# hemispheres both come from a stochastic block model, which models the probability
# of any potential edge as a function of the groups that the source and target nodes
# are part of.
#
# For now, we use some broad cell type categorizations for each neuron to determine its
# group. Alternatively, there are many methods for *estimating* these assignments to
# groups for each neuron, which we do not explore here.

#%%
import datetime
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import merge_axes, rotate_labels
from graspologic.plot import heatmap, networkplot
from graspologic.simulations import sbm
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, savefig
from pkg.perturb import remove_edges
from pkg.plot import (
    bound_points,
    compare_probability_row,
    heatmap_grouped,
    networkplot_grouped,
    set_theme,
)
from pkg.plot.utils import make_sequential_colormap
from pkg.stats import stochastic_block_test
from seaborn.utils import relative_luminance
from svgutils.compose import SVG, Figure, Panel, Text

DISPLAY_FIGS = False

FILENAME = "sbm_unmatched_test"


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
rng = np.random.default_rng(8888)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()
neutral_color = sns.color_palette("Set2")[2]

GROUP_KEY = "simple_group"

left_adj, left_nodes = load_unmatched(side="left")
right_adj, right_nodes = load_unmatched(side="right")

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values

#%% [markdown]
# ## The stochastic block model (SBM)
# A [**stochastic block model (SBM)**
# ](https://en.wikipedia.org/wiki/Stochastic_block_model)
# is a popular statistical model of networks. Put simply, this model treats the
# probability of an edge occuring between node $i$ and node $j$ as purely a function of
# the *communities* or *groups* that node $i$ and $j$ belong to. Therefore, this model
# is parameterized by:
#
#    1. An assignment of each node in the network to a group. Note that this assignment
#       can be considered to be deterministic or random, depending on the specific
#       framing of the model one wants to use.
#    2. A set of group-to-group connection probabilities
#
# ```{admonition} Math
# Let $n$ be the number of nodes, and $K$ be the number of groups in an SBM. For a
# network $A$ sampled from an SBM:

# $$ A \sim SBM(B, \tau)$$

# We say that for all $(i,j), i \neq j$, with $i$ and $j$ both running
# from $1 ... n$ the probability of edge $(i,j)$ occuring is:

# $$ P[A_{ij} = 1] = P_{ij} = B_{\tau_i, \tau_j} $$

# where $B \in [0,1]^{K \times K}$ is a matrix of group-to-group connection
# probabilities and $\tau \in \{1...K\}^n$ is a vector of node-to-group assignments.
# Note that here we are assuming $\tau$ is a fixed vector of assignments, though other
# formuations of the SBM allow these assignments to themselves come from a categorical
# distribution.
# ```

#%% [markdown]
# ## Testing under the SBM model
# Assuming this model, there are a few ways that one could test for differences between
# two networks. In our case, we are interested in comparing the group-to-group
# connection probability matrices, $B$,  for the left and right hemispheres.

# ````{admonition} Math
# We are interested in testing:

# ```{math}
# :label: sbm_unmatched_null
# H_0: B^{(L)} = B^{(R)}, \quad H_A: B^{(L)} \neq B^{(R)}
# ```
#
# ````
#
# Rather than having to compare one proportion as in [](er_unmatched_test.ipynb), we are
# now interedted in comparing all $K^2$ probabilities between the SBM models for the
# left and right hemispheres.

# ```{admonition} Math
# The hypothesis test above can be decomposed into $K^2$ indpendent hypotheses.
# $B^{(L)}$
# and $B^{(R)}$ are both $K \times K$ matrices, where each element $b_{kl}$ represents
# the probability of a connection from a neuron in group $k$ to one in group $l$. We
# also know that group $k$ for the left network corresponds with group $k$ for the
# right. In other words, the *groups* are matched. Thus, we are interested in testing,
# for $k, l$ both running from $1...K$:

# $$ H_0: B_{kl}^{(L)} = B_{kl}^{(R)},
# \quad H_A: B_{kl}^{(L)} \neq B_{kl}^{(R)}$$

# ```
#
# Thus, we will use
# [Fisher's exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) to
# compare each set of probabilities. To combine these multiple hypotheses into one, we
# will use [Fisher's method](https://en.wikipedia.org/wiki/Fisher%27s_method) for
# combining p-values to give us a p-value for the overall test. We also can look at
# the p-values for each of the individual tests after correction for multiple
# comparisons by the
# [Bonferroni-Holm method.
# ](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)

#%% [markdown]
# For the current investigation, we focus on the case where $\tau$ is known ahead of
# time, sometimes called the **A priori SBM**. We use some broad cell type labels which
# were described in the paper which published the data to
# define the group assignments $\tau$. Here, we do not explore
# estimating these assignments, though many techniques exist for doing so. We note that
# the results presented here could change depending on the group assignments which are
# used. We also do not consider tests which would compare the assignment vectors,
# $\tau$. {numref}`Figure {number} <fig:sbm_unmatched_test-group_counts>` shows the
# number of neurons in each group in the group assignments $\tau$ for the left and
# the right hemispheres. The number of neurons in each group is quite similar between
# the two hemispheres.

#%%


np.random.seed(888888)
ns = [5, 6, 7]
B = np.array([[0.8, 0.2, 0.05], [0.05, 0.9, 0.2], [0.05, 0.05, 0.7]])
A1, labels = sbm(ns, B, directed=True, loops=False, return_labels=True)
A2 = sbm(ns, B, directed=True, loops=False)

node_data = pd.DataFrame(index=np.arange(A1.shape[0]))
node_data["labels"] = labels + 1
palette = dict(zip(np.unique(labels) + 1, sns.color_palette("Set2")[3:]))

fig, axs = plt.subplots(2, 4, figsize=(16, 6))
ax = axs[0, 0]
networkplot_grouped(A1, node_data, palette=palette, ax=ax)
ax.set_title("Group neurons\nby cell type", fontsize="medium")
ax.set_ylabel(
    "Left",
    color=network_palette["Left"],
    size="large",
    rotation=0,
    ha="right",
    labelpad=10,
)

ax = axs[1, 0]
networkplot_grouped(A2, node_data, palette=palette, ax=ax)
ax.set_ylabel(
    "Right",
    color=network_palette["Right"],
    size="large",
    rotation=0,
    ha="right",
    labelpad=10,
)

ax = axs[0, 1]
_, _, misc = stochastic_block_test(A1, A1, node_data["labels"], node_data["labels"])
Bhat1 = misc["probabilities1"].values
top_ax = heatmap_grouped(Bhat1, [1, 2, 3], palette=palette, ax=ax)
top_ax.set_title("Fit stochastic\nblock models", fontsize="medium")

ax = axs[1, 1]
_, _, misc = stochastic_block_test(A2, A2, node_data["labels"], node_data["labels"])
Bhat2 = misc["probabilities1"].values
heatmap_grouped(Bhat2, [1, 2, 3], palette=palette, ax=ax)


cmap = make_sequential_colormap("Blues")

ax = merge_axes(fig, axs, rows=None, cols=2)
ax.set(xlim=(0, 1), ylim=(0.18, 1))

ax.text(0.4, 0.93, r"$\hat{B}^{(L)}$", color=network_palette["Left"])
ax.text(0.8, 0.93, r"$\hat{B}^{(R)}$", color=network_palette["Right"])
compare_probability_row(1, 1, Bhat1, Bhat2, 0.9, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(1, 2, Bhat1, Bhat2, 0.85, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(1, 3, Bhat1, Bhat2, 0.8, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(2, 1, Bhat1, Bhat2, 0.75, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(2, 2, Bhat1, Bhat2, 0.7, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(2, 3, Bhat1, Bhat2, 0.65, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(3, 1, Bhat1, Bhat2, 0.6, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(3, 2, Bhat1, Bhat2, 0.55, cmap=cmap, palette=palette, ax=ax)
compare_probability_row(3, 3, Bhat1, Bhat2, 0.5, cmap=cmap, palette=palette, ax=ax)


ax.annotate(
    "",
    xy=(0.645, 0.48),
    xytext=(0.5, 0.41),
    arrowprops=dict(
        arrowstyle="simple",
        facecolor="black",
    ),
)


def get_text_points(text, transformer, renderer):
    bbox = text.get_window_extent(renderer=renderer)
    bbox_points = bbox.get_points()
    out_points = transformer.transform(bbox_points)
    return out_points


def get_text_width(text, transformer, renderer):
    points = get_text_points(text, transformer, renderer)
    width = points[1][0] - points[0][0]
    return width


def multicolor_text(x, y, texts, colors, ax=None, space_scale=1.0, **kwargs):
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    transformer = ax.transData.inverted()

    # make this dummy text to get proper space width, then delete
    text = ax.text(0.5, 0.5, " ")
    space_width = get_text_width(text, transformer, renderer)
    space_width *= space_scale
    text.remove()

    # TODO make the spacing "smart"
    text_objs = []
    for text, color in zip(texts, colors):
        text_obj = ax.text(x, y, text, color=color, **kwargs)
        text_width = get_text_width(text_obj, transformer, renderer)
        x += text_width
        x += space_width
        text_objs.append(text_obj)

    return text_objs


def get_texts_points(texts, ax=None):
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    transformer = ax.transData.inverted()

    x_maxs = []
    x_mins = []
    y_maxs = []
    y_mins = []
    for text in texts:
        points = get_text_points(text, transformer, renderer)
        x_maxs.append(points[1][0])
        x_mins.append(points[0][0])
        y_maxs.append(points[1][1])
        y_mins.append(points[0][1])

    x_max = max(x_maxs)
    x_min = min(x_mins)
    y_max = max(y_maxs)
    y_min = min(y_mins)
    return x_min, x_max, y_min, y_max


def bound_texts(texts, ax=None, xpad=0, ypad=0, **kwargs):
    x_min, x_max, y_min, y_max = get_texts_points(texts, ax=ax)
    xy = (x_min - xpad, y_min - ypad)
    width = x_max - x_min + 2 * xpad
    height = y_max - y_min + 2 * ypad
    patch = mpl.patches.Rectangle(xy=xy, width=width, height=height, **kwargs)
    ax.add_patch(patch)


y = 0.34
texts = multicolor_text(
    0.2,
    y,
    [r"$H_0$:", r"$\hat{B}^{L}_{ij}$", r"$=$", r"$\hat{B}^{R}_{ij}$"],
    ["black", network_palette["Left"], "black", network_palette["Right"]],
    fontsize="large",
    ax=ax,
)

y = y - 0.1
texts += multicolor_text(
    0.2,
    y,
    [r"$H_A$:", r"$\hat{B}^{L}_{ij}$", r"$\neq$", r"$\hat{B}^{R}_{ij}$"],
    ["black", network_palette["Left"], "black", network_palette["Right"]],
    fontsize="large",
    ax=ax,
)

bound_texts(
    texts,
    ax=ax,
    xpad=0.03,
    ypad=0.01,
    facecolor="white",
    edgecolor="lightgrey",
)


# patch = mpl.patches.Rectangle(
#     xy=(0.18, y - 0.03),
#     width=0.7,
#     height=0.21,
#     facecolor="white",
#     edgecolor="lightgrey",
# )
# ax.add_patch(patch)


ax.set_title("Compare estimated\nprobabilities", fontsize="medium")

ax.axis("off")

ax = merge_axes(fig, axs, rows=None, cols=3)
ax.axis("off")
ax.set_title("Combine p-values\nfor overall test", fontsize="medium")

ax.plot([0, 0.4], [0.9, 0.7], color="black")
ax.plot([0, 0.4], [0.5, 0.7], color="black")
ax.set(xlim=(0, 1), ylim=(0.18, 1))
ax.text(0.42, 0.7, r"$p = ...$", va="center", ha="left")

ax.annotate(
    "",
    xy=(0.64, 0.68),
    xytext=(0.5, 0.41),
    arrowprops=dict(
        arrowstyle="simple",
        facecolor="black",
    ),
)

y = 0.34
texts = multicolor_text(
    0.2,
    y,
    [r"$H_0$:", r"$\hat{B}^{L}$", r"$=$", r"$\hat{B}^{R}$"],
    ["black", network_palette["Left"], "black", network_palette["Right"]],
    fontsize="large",
    ax=ax,
)

y = y - 0.1
texts += multicolor_text(
    0.2,
    y,
    [r"$H_A$:", r"$\hat{B}^{L}$", r"$\neq$", r"$\hat{B}^{R}$"],
    ["black", network_palette["Left"], "black", network_palette["Right"]],
    fontsize="large",
    ax=ax,
)

bound_texts(
    texts,
    ax=ax,
    xpad=0.03,
    ypad=0.01,
    facecolor="white",
    edgecolor="lightgrey",
)

fig.set_facecolor("w")
gluefig("sbm_methods_explain", fig)


#%%

stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    combine_method="tippett",
)
glue("uncorrected_pvalue", pvalue)
n_tests = misc["n_tests"]
glue("n_tests", n_tests)
print(f"{pvalue:.2g}")

#%%
min_stat, min_pvalue, min_misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    combine_method="min",
)
glue("uncorrected_pvalue_min", pvalue)
print(min_pvalue)

#%%
set_theme(font_scale=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

group_counts_left = misc["group_counts1"]
group_counts_right = misc["group_counts2"]

for i in range(len(group_counts_left)):
    ax.bar(i - 0.17, group_counts_left[i], width=0.3, color=network_palette["Left"])
    ax.bar(i + 0.17, group_counts_right[i], width=0.3, color=network_palette["Right"])

rotate_labels(ax)
ax.set(
    ylabel="Count",
    xlabel="Group",
    xticks=np.arange(len(group_counts_left)) + 0.2,
    xticklabels=group_counts_left.index,
)
gluefig("group_counts", fig)

#%%

#%% [markdown]

# ```{glue:figure} fig:sbm_unmatched_test-group_counts
# :name: "fig:sbm_unmatched_test-group_counts"

# The number of neurons in each group in each hemisphere. Note the similarity between
# the hemispheres.
# ```

#%%


def shrink_axis(ax, scale=0.7):
    pos = ax.get_position()
    mid = (pos.ymax + pos.ymin) / 2
    height = pos.ymax - pos.ymin
    new_pos = Bbox(
        [
            [pos.xmin, mid - scale * 0.5 * height],
            [pos.xmax, mid + scale * 0.5 * height],
        ]
    )
    ax.set_position(new_pos)


def plot_stochastic_block_test(misc, pvalue_vmin=None, annot_missing=True):
    # get values
    B1 = misc["probabilities1"]
    B2 = misc["probabilities2"]
    null_odds = misc["null_odds"]
    B2 = B2 * null_odds

    index = B1.index
    p_max = max(B1.values.max(), B2.values.max())
    uncorrected_pvalues = misc["uncorrected_pvalues"]
    n_tests = misc["n_tests"]
    K = B1.shape[0]
    alpha = 0.05
    hb_thresh = alpha / n_tests

    # set up plot
    pad = 2
    width_ratios = [0.5, pad + 0.8, 10, pad, 10]
    set_theme(font_scale=1.25)
    fig, axs = plt.subplots(
        1,
        len(width_ratios),
        figsize=(20, 10),
        gridspec_kw=dict(
            width_ratios=width_ratios,
        ),
    )
    left_col = 2
    right_col = 4
    # pvalue_col = 6

    heatmap_kws = dict(
        cmap="Blues", square=True, cbar=False, vmax=p_max, fmt="s", xticklabels=True
    )

    # heatmap of left connection probabilities
    annot = np.full((K, K), "")
    annot[B1.values == 0] = 0
    ax = axs[left_col]
    sns.heatmap(B1, ax=ax, annot=annot, **heatmap_kws)
    ax.set(ylabel="Source group", xlabel="Target group")
    ax.set_title(r"$\hat{B}$ left", fontsize="xx-large", color=network_palette["Left"])

    # heatmap of right connection probabilities
    annot = np.full((K, K), "")
    annot[B2.values == 0] = 0
    ax = axs[right_col]
    im = sns.heatmap(B2, ax=ax, annot=annot, **heatmap_kws)
    ax.set(ylabel="", xlabel="Target group")
    text = r"$\hat{B}$ right"
    if null_odds != 1:
        text = r"$c$" + text
    ax.set_title(text, fontsize="xx-large", color=network_palette["Right"])
    # ax.set(yticks=[], yticklabels=[])

    # handle the colorbars
    # NOTE: did it this way cause the other options weren't playing nice with auto
    # constrain
    # layouts.

    ax = axs[0]
    shrink_axis(ax, scale=0.5)
    _ = fig.colorbar(
        im.get_children()[0],
        cax=ax,
        fraction=1,
        shrink=1,
        ticklocation="left",
    )
    ax.set_title("Estimated\nprobability")

    # plot p-values
    # ax = axs[pvalue_col]

    # if annot_missing:
    #     annot = np.full((K, K), "")
    #     annot[(B1.values == 0) & (B2.values == 0)] = "B"
    #     annot[(B1.values == 0) & (B2.values != 0)] = "L"
    #     annot[(B1.values != 0) & (B2.values == 0)] = "R"
    # else:
    #     annot = False
    # plot_pvalues = np.log10(uncorrected_pvalues)
    # plot_pvalues[np.isnan(plot_pvalues)] = 0
    # im = sns.heatmap(
    #     plot_pvalues,
    #     ax=ax,
    #     annot=annot,
    #     cmap="RdBu",
    #     center=0,
    #     square=True,
    #     cbar=False,
    #     fmt="s",
    #     vmin=pvalue_vmin,
    # )
    # ax.set(ylabel="", xlabel="Target group")
    # ax.set(xticks=np.arange(K) + 0.5, xticklabels=index)
    # ax.set_title(r"$log_{10}($p-value$)$", fontsize="xx-large")

    # colors = im.get_children()[0].get_facecolors()
    # significant = uncorrected_pvalues < hb_thresh

    # # NOTE: the x's looked bad so I did this super hacky thing...
    # pad = 0.2
    # for idx, (is_significant, color) in enumerate(
    #     zip(significant.values.ravel(), colors)
    # ):
    #     if is_significant:
    #         i, j = np.unravel_index(idx, (K, K))
    #         # REF: seaborn heatmap
    #         lum = relative_luminance(color)
    #         text_color = ".15" if lum > 0.408 else "w"

    #         xs = [j + pad, j + 1 - pad]
    #         ys = [i + pad, i + 1 - pad]
    #         ax.plot(xs, ys, color=text_color, linewidth=4)
    #         xs = [j + 1 - pad, j + pad]
    #         ys = [i + pad, i + 1 - pad]
    #         ax.plot(xs, ys, color=text_color, linewidth=4)

    # # plot colorbar for the pvalue plot
    # # NOTE: only did it this way for consistency with the other colorbar
    # ax = axs[7]
    # shrink_axis(ax, scale=0.5)
    # _ = fig.colorbar(
    #     im.get_children()[0],
    #     cax=ax,
    #     fraction=1,
    #     shrink=1,
    #     ticklocation="right",
    # )

    # # fig.text(0.11, 0.85, "A)", fontweight="bold", fontsize=50)
    # # fig.text(0.63, 0.85, "B)", fontweight="bold", fontsize=50)

    # remove dummy axes
    for i in range(len(width_ratios)):
        if not axs[i].has_data():
            axs[i].set_visible(False)

    return fig, axs


fig, axs = plot_stochastic_block_test(misc)

gluefig("sbm_uncorrected", fig)

# need to save this for later for setting colorbar the same on other plot
pvalue_vmin = np.log10(np.nanmin(misc["uncorrected_pvalues"].values))
glue("pvalue_vmin", pvalue_vmin)

#%%


def plot_pvalues(
    ax, cax, uncorrected_pvalues, B1, B2, hb_thresh, pvalue_vmin, annot_missing=True
):
    K = len(B1)
    index = B1.index
    if annot_missing:
        annot = np.full((K, K), "")
        annot[(B1.values == 0) & (B2.values == 0)] = "B"
        annot[(B1.values == 0) & (B2.values != 0)] = "L"
        annot[(B1.values != 0) & (B2.values == 0)] = "R"
    else:
        annot = False
    plot_pvalues = np.log10(uncorrected_pvalues)
    plot_pvalues[np.isnan(plot_pvalues)] = 0
    im = sns.heatmap(
        plot_pvalues,
        ax=ax,
        annot=annot,
        cmap="RdBu",
        center=0,
        square=True,
        cbar=False,
        fmt="s",
        vmin=pvalue_vmin,
    )
    ax.set(ylabel="Source group", xlabel="Target group")
    ax.set(xticks=np.arange(K) + 0.5, xticklabels=index)
    ax.set_title(r"Probability comparison", fontsize="x-large")

    colors = im.get_children()[0].get_facecolors()
    significant = uncorrected_pvalues < hb_thresh

    # NOTE: the x's looked bad so I did this super hacky thing...
    pad = 0.2
    for idx, (is_significant, color) in enumerate(
        zip(significant.values.ravel(), colors)
    ):
        if is_significant:
            i, j = np.unravel_index(idx, (K, K))
            # REF: seaborn heatmap
            lum = relative_luminance(color)
            text_color = ".15" if lum > 0.408 else "w"

            xs = [j + pad, j + 1 - pad]
            ys = [i + pad, i + 1 - pad]
            ax.plot(xs, ys, color=text_color, linewidth=4)
            xs = [j + 1 - pad, j + pad]
            ys = [i + pad, i + 1 - pad]
            ax.plot(xs, ys, color=text_color, linewidth=4)

    # plot colorbar for the pvalue plot
    # NOTE: only did it this way for consistency with the other colorbar
    shrink_axis(cax, scale=0.5)
    _ = fig.colorbar(
        im.get_children()[0],
        cax=cax,
        fraction=1,
        shrink=1,
        ticklocation="left",
    )
    cax.set_title(r"$log_{10}$" + "\np-value", pad=20)


width_ratios = [0.5, 3, 10]
fig, axs = plt.subplots(
    1,
    3,
    figsize=(10, 10),
    gridspec_kw=dict(
        width_ratios=width_ratios,
    ),
)
axs[1].remove()
plot_pvalues(
    axs[2],
    axs[0],
    misc["uncorrected_pvalues"],
    misc["probabilities1"],
    misc["probabilities2"],
    0.05 / misc["n_tests"],
    pvalue_vmin,
    annot_missing=False,
)
gluefig("sbm_uncorrected_pvalues", fig)


#%% [markdown]
# Next, we run the test for bilateral symmetry under the stochastic block model.
# {numref}`Figure {number} <fig:sbm_unmatched_test-sbm_uncorrected>` shows both the
# estimated group-to-group probability matrices, $\hat{B}^{(L)}$ and $\hat{B}^{(R)}$,
# as well as the p-values from each test comparing each element of these matrices. From
# a visual comparison of $\hat{B}^{(L)}$ and $\hat{B}^{(R)}$
# {numref}`(Figure {number} A) <fig:sbm_unmatched_test-sbm_uncorrected>`, we see that
# the
# group-to-group connection probabilities are qualitatively similar. Note also that some
# group-to-group connection probabilities are zero, making it non-sensical to do a
# comparision of binomial proportions. We highlight these elements in the $\hat{B}$
# matrices with an explicit "0", noting that we did not run the corresponding test in
# these cases.
#
# In {numref}`Figure {number} B <fig:sbm_unmatched_test-sbm_uncorrected>`, we see the
# p-values from all {glue:text}`sbm_unmatched_test-n_tests` that were run. After
# Bonferroni-Holm correction, 5 tests yield p-values less than 0.05, indicating that
# we reject the null hypothesis that those elements of the $\hat{B}$ matrices are the
# same between the two hemispheres. We also combine all p-values using Fisher's method,
# which yields an overall p-value for the entire null hypothesis in
# Equation {eq}`sbm_unmatched_null` of
# {glue:text}`sbm_unmatched_test-uncorrected_pvalue:0.2e`.
#
# ```{glue:figure} fig:sbm_unmatched_test-sbm_uncorrected
# :name: "fig:sbm_unmatched_test-sbm_uncorrected"
#
# Comparison of stochastic block model fits for the left and right hemispheres.
# **A)** The estimated group-to-group connection probabilities for the left
# and right hemispheres appear qualitatively similar. Any estimated
# probabilities which are zero (i.e. no edge was present between a given pair of
# communities) is indicated explicitly with a "0" in that cell of the matrix.
# **B)** The p-values for each hypothesis test between individual elements of
# the block probability matrices. In other words, each cell represents a test for
# whether a given group-to-group connection probability is the same on the left and the
# right sides. "X" denotes a significant p-value after Bonferroni-Holm correction,
# with $\alpha=0.05$. "B" indicates that a test was not run since the estimated
# probability
# was zero in that cell on both the left and right. "L" indicates this was the case on
# the left only, and "R" that it was the case on the right only. These individual
# p-values were combined using Fisher's method, resulting in an overall p-value (for the
# null hypothesis that the two group connection probability matrices are the same) of
# {glue:text}`sbm_unmatched_test-uncorrected_pvalue:0.2e`.
# ```

#%% [markdown]
# ## Adjusting for a difference in density
# From {numref}`Figure {number} <fig:sbm_unmatched_test-sbm_uncorrected>`, we see that
# we have sufficient evidence to reject
# the null hypothesis of bilateral symmetry under this version of the SBM. However,
# we already saw in [](er_unmatched_test) that the overall
# densities between the two networks are different. Could it be that this rejection of
# the null hypothesis under the SBM can be explained purely by this difference in
# density? In other words, are the group-to-group connection probabilities on the right
# simply a "scaled up" version of those on the right, where each probability is scaled
# by the same amount?
#
# In {numref}`Figure {number} <fig:sbm_unmatched_test-probs_uncorrected>`,
# we plot the estimated
# probabilities on the left and the right hemispheres (i.e. each element of $\hat{B}$),
# as
# well as the difference between them. While subtle, we note that there is a slight
# tendency for the left hemisphere estimated probability to be lower than the
# corresponding one on the right. Specifically, we can also look at the group-to-group
# connection probabilities which were significantly different in
# {numref}`Figure {number} <fig:sbm_unmatched_test-sbm_uncorrected>` - these are plotted
# in {numref}`Figure {number} <fig:sbm_unmatched_test-significant_p_comparison>`. Note
# that in every case, the estimated probability on the right is higher with that on the
# right.
#


#%%


def plot_estimated_probabilities(misc):
    B1 = misc["probabilities1"]
    B2 = misc["probabilities2"]
    null_odds = misc["null_odds"]
    B2 = B2 * null_odds
    B1_ravel = B1.values.ravel()
    B2_ravel = B2.values.ravel()
    arange = np.arange(len(B1_ravel))
    sum_ravel = B1_ravel + B2_ravel
    sort_inds = np.argsort(-sum_ravel)
    B1_ravel = B1_ravel[sort_inds]
    B2_ravel = B2_ravel[sort_inds]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax = axs[0]
    sns.scatterplot(
        x=arange,
        y=B1_ravel,
        color=network_palette["Left"],
        ax=ax,
        linewidth=0,
        s=15,
        alpha=0.5,
    )
    sns.scatterplot(
        x=arange,
        y=B2_ravel,
        color=network_palette["Right"],
        ax=ax,
        linewidth=0,
        s=15,
        alpha=0.5,
        zorder=-1,
    )
    ax.text(
        0.7,
        0.8,
        "Left",
        color=network_palette["Left"],
        transform=ax.transAxes,
    )
    ax.text(
        0.7,
        0.7,
        "Right",
        color=network_palette["Right"],
        transform=ax.transAxes,
    )
    ax.set_yscale("log")
    ax.set(
        ylabel="Estimated probability " + r"($\hat{p}$)",
        xticks=[],
        xlabel="Sorted group pairs",
    )
    ax.spines["bottom"].set_visible(False)

    ax = axs[1]
    diff = B1_ravel - B2_ravel
    yscale = np.max(np.abs(diff))
    yscale *= 1.05
    sns.scatterplot(
        x=arange, y=diff, ax=ax, linewidth=0, s=25, color=neutral_color, alpha=1
    )
    ax.axhline(0, color="black", zorder=-1)
    ax.spines["bottom"].set_visible(False)
    ax.set(
        xticks=[],
        ylabel=r"$\hat{p}_{left} - \hat{p}_{right}$",
        xlabel="Sorted group pairs",
        ylim=(-yscale, yscale),
    )
    n_greater = np.count_nonzero(diff > 0)
    n_total = len(diff)
    ax.text(
        0.3,
        0.8,
        f"Left connection stronger ({n_greater}/{n_total})",
        color=network_palette["Left"],
        transform=ax.transAxes,
    )
    n_lesser = np.count_nonzero(diff < 0)
    ax.text(
        0.3,
        0.15,
        f"Right connection stronger ({n_lesser}/{n_total})",
        color=network_palette["Right"],
        transform=ax.transAxes,
    )

    fig.text(0.02, 0.905, "A)", fontweight="bold", fontsize=30)
    fig.text(0.02, 0.49, "B)", fontweight="bold", fontsize=30)

    return fig, ax


fig, ax = plot_estimated_probabilities(misc)
gluefig("probs_uncorrected", fig)


#%% [markdown]
# ```{glue:figure} fig:sbm_unmatched_test-probs_uncorrected
# :name: "fig:sbm_unmatched_test-probs_uncorrected"

# Comparison of estimated connection probabilities for the left and right hemispheres.
# **A)** The estimated group-to-group connection probabilities ($\hat{p}$), sorted by
# the mean left/right connection probability. Note the very subtle tendency for the
# left probability to be lower than the corresponding one on the right. **B)** The
# differences between corresponding group-to-group connection probabilities
# ($\hat{p}^{(L)} - \hat{p}^{(R)}$). The trend of the left connection probabilities
# being slightly smaller than the corresponding probability on the right is more
# apparent here, as there are more negative than positive values.
# ```

#%%
# B1 = misc["probabilities1"]
# B2 = misc["probabilities2"]
# null_odds = misc["null_odds"]
# B2 = B2 * null_odds
# B1_ravel = B1.values.ravel()
# B2_ravel = B2.values.ravel()

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# sns.scatterplot(x=B2_ravel, y=B1_ravel, color=neutral_color)
# ax.set(yscale="log", xscale="log")
# ax.set_xlabel("Right connection probability", color=network_palette["Right"])
# ax.set_ylabel("Left connection probability", color=network_palette["Left"])
# ax.plot([0.00001, 1], [0.00001, 1], color="black", linestyle="--")


#%%


def plot_significant_probabilities(misc):
    B1 = misc["probabilities1"]
    B2 = misc["probabilities2"]
    null_odds = misc["null_odds"]
    B2 = B2 * null_odds
    index = B1.index
    uncorrected_pvalues = misc["uncorrected_pvalues"]
    n_tests = misc["n_tests"]

    alpha = 0.05
    hb_thresh = alpha / n_tests
    significant = uncorrected_pvalues < hb_thresh

    row_inds, col_inds = np.nonzero(significant.values)

    rows = []
    for row_ind, col_ind in zip(row_inds, col_inds):
        source = index[row_ind]
        target = index[col_ind]
        left_p = B1.loc[source, target]
        right_p = B2.loc[source, target]
        pair = source + r"$\rightarrow$" + "\n" + target
        rows.append(
            {
                "source": source,
                "target": target,
                "p": left_p,
                "side": "Left",
                "pair": pair,
            }
        )
        rows.append(
            {
                "source": source,
                "target": target,
                "p": right_p,
                "side": "Right",
                "pair": pair,
            }
        )
    sig_data = pd.DataFrame(rows)

    mean_ps = sig_data.groupby("pair")["p"].mean()
    pair_orders = mean_ps.sort_values(ascending=False).index

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.pointplot(
        data=sig_data,
        y="p",
        x="pair",
        ax=ax,
        hue="side",
        hue_order=["Right", "Left"],
        order=pair_orders,
        dodge=False,
        join=False,
        palette=network_palette,
        markers="_",
        scale=1.75,
    )
    ax.tick_params(axis="x", length=7)

    ax.set_yscale("log")
    ax.set_ylim((0.01, 1))

    leg = ax.get_legend()
    leg.set_title("Hemisphere")
    leg.set_frame_on(True)
    rotate_labels(ax)
    ax.set(xlabel="Group pair", ylabel="Connection probability")
    return fig, ax


fig, ax = plot_significant_probabilities(misc)
gluefig("significant_p_comparison", fig)


#%%

#%% [markdown]
# ```{glue:figure} fig:sbm_unmatched_test-significant_p_comparison
# :name: "fig:sbm_unmatched_test-significant_p_comparison"
#
# Comparison of estimated group-to-group connection probabilities for the group-pairs
# which were significantly different in
# {numref}`Figure {number} <fig:sbm_unmatched_test-sbm_uncorrected>`.
# In each case, the connection probability on the right hemisphere is higher.
# ```

#%%
total_width = 1000
total_height = 1500

FIG_PATH = FIG_PATH / FILENAME

fontsize = 35

sbm_methods_explain_svg = SVG(FIG_PATH / "sbm_methods_explain.svg")
sbm_methods_explain_svg_scaler = 1 / sbm_methods_explain_svg.height * total_height / 4
sbm_methods_explain_svg = sbm_methods_explain_svg.scale(sbm_methods_explain_svg_scaler)

sbm_methods_explain = Panel(
    sbm_methods_explain_svg,
    Text("A)", 5, 20, size=fontsize, weight="bold"),
)

sbm_uncorrected_svg = SVG(FIG_PATH / "sbm_uncorrected.svg")
sbm_uncorrected_svg.scale(1 / sbm_uncorrected_svg.height * total_height / 3)
sbm_uncorrected = Panel(
    sbm_uncorrected_svg, Text("B)", 5, 20, size=fontsize, weight="bold")
).move(0, 330)

sbm_uncorrected_pvalues = SVG(FIG_PATH / "sbm_uncorrected_pvalues.svg")
sbm_uncorrected_pvalues.scale(1 / sbm_uncorrected_pvalues.height * total_height / 3)
sbm_uncorrected_pvalues = Panel(
    sbm_uncorrected_pvalues, Text("C)", 5, 0, size=fontsize, weight="bold")
).move(0, 750)

significant_p_comparison = SVG(FIG_PATH / "significant_p_comparison.svg")
significant_p_comparison.scale(
    1 / significant_p_comparison.height * total_height / 3
).scale(0.9)
significant_p_comparison = Panel(
    significant_p_comparison.move(20, 20),
    Text("D)", 0, 0, size=fontsize, weight="bold"),
).move(475, 750)

fig = Figure(
    810,
    1170,
    sbm_methods_explain,
    sbm_uncorrected,
    sbm_uncorrected_pvalues,
    significant_p_comparison,
)
fig.save(FIG_PATH / "sbm_uncorrected_composite.svg")
fig


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
