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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import merge_axes, rotate_labels
from graspologic.simulations import sbm
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, savefig
from pkg.plot import (
    bound_texts,
    compare_probability_row,
    heatmap_grouped,
    multicolor_text,
    networkplot_simple,
    plot_pvalues,
    plot_stochastic_block_probabilities,
    set_theme,
)
from pkg.plot.utils import make_sequential_colormap, shrink_axis
from pkg.stats import stochastic_block_test
from pkg.utils import get_toy_palette, sample_toy_networks
from pkg.utils.toy import sample_toy_networks
from svgutils.compose import SVG, Figure, Panel, Text
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
import matplotlib.transforms as mtrans
from matplotlib.font_manager import FontProperties
from pkg.plot import draw_hypothesis_box

from pkg.plot import shrink_axis


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

A1, A2, node_data = sample_toy_networks()
palette = get_toy_palette()

fig, axs = plt.subplots(
    2, 4, figsize=(14, 6), gridspec_kw=dict(wspace=0, hspace=0), constrained_layout=True
)

ax = axs[0, 0]
networkplot_simple(A1, node_data, palette=palette, ax=ax, group=True)
ax.set_title("Group neurons", fontsize="medium")
ax.set_ylabel(
    "Left",
    color=network_palette["Left"],
    size="large",
    rotation=0,
    ha="right",
    labelpad=10,
)

ax = axs[1, 0]
networkplot_simple(A2, node_data, palette=palette, ax=ax, group=True)
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
top_ax.set_title(r"$\hat{B}^{(L)}$", color=network_palette["Left"])
# shrink_axis(ax, scale=0.8)

ax.set_title("Estimate group-to-group\nconnection probabilities", pad=20)

# top_ax.set_title("Fit stochastic\nblock models", fontsize="medium")

ax = axs[1, 1]
_, _, misc = stochastic_block_test(A2, A2, node_data["labels"], node_data["labels"])
Bhat2 = misc["probabilities1"].values
top_ax = heatmap_grouped(Bhat2, [1, 2, 3], palette=palette, ax=ax)
top_ax.set_title(r"$\hat{B}^{(R)}$", color=network_palette["Right"])
# shrink_axis(ax, scale=0.8)

ax.set_title("")

cmap = make_sequential_colormap("Blues")


ax = merge_axes(fig, axs, rows=None, cols=2)
ax.set(xlim=(0, 13), ylim=(0, 1))


def compare_probability_row(i, j, y, Bhat1, Bhat2, ax=None, palette=None):
    prob1 = Bhat1[i, j]
    prob2 = Bhat2[i, j]

    linestyle_kws = dict(linewidth=1, color="dimgrey")
    ax.axvline(4, ymin=0.32, ymax=0.89, **linestyle_kws)
    ax.axvline(9, ymin=0.32, ymax=0.89, **linestyle_kws)
    ax.plot([4, 9], [0.89, 0.89], **linestyle_kws)
    ax.plot([4, 9], [0.32, 0.32], **linestyle_kws)

    ax.text(
        5.5,
        0.9,
        r"$\hat{B}^{(L)}_{ij}$",
        ha="center",
        va="bottom",
        color=network_palette["Left"],
    )

    ax.text(
        8.5,
        0.9,
        r"$\hat{B}^{(R)}_{ij}$",
        ha="center",
        va="bottom",
        color=network_palette["Right"],
    )

    ax.plot(
        [1],
        [y],
        "o",
        markersize=13,
        markeredgecolor="black",
        markeredgewidth=1,
        markerfacecolor=palette[i + 1],
    )
    ax.plot(
        [3],
        [y],
        "o",
        markersize=13,
        markeredgecolor="black",
        markeredgewidth=1,
        markerfacecolor=palette[j + 1],
    )
    ax.annotate(
        "",
        xy=(3, y),
        xytext=(1, y),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="angle,angleA=60,angleB=-60,rad=30",
            facecolor="black",
            shrinkA=8,
            shrinkB=7,
            # mutation_scale=,
        ),
    )

    ax.plot([5], [y], "s", markersize=15, color=cmap(prob1))
    ax.plot([8], [y], "s", markersize=15, color=cmap(prob2))
    ax.text(6.5, y, r"$\overset{?}{=}$", fontsize="large", va="center", ha="center")
    ax.text(10, y, r"$\rightarrow$", fontsize="large", va="center", ha="center")
    p_text = r"$p_{"
    p_text += str(i + 1)
    p_text += str(j + 1)
    p_text += r"}$"
    ax.text(12, y - 0.01, p_text, fontsize="large", va="center", ha="center")
    ax.set(xticks=[], yticks=[])


compare_probability_row(0, 0, 0.83, Bhat1, Bhat2, ax=ax, palette=palette)
compare_probability_row(0, 1, 0.72, Bhat1, Bhat2, ax=ax, palette=palette)

ax.plot(
    [6.65], [0.64], ".", markersize=4, markeredgecolor="black", markerfacecolor="black"
)
ax.plot(
    [6.65], [0.61], ".", markersize=4, markeredgecolor="black", markerfacecolor="black"
)
ax.plot(
    [6.65], [0.58], ".", markersize=4, markeredgecolor="black", markerfacecolor="black"
)

compare_probability_row(2, 1, 0.48, Bhat1, Bhat2, ax=ax, palette=palette)
compare_probability_row(2, 2, 0.37, Bhat1, Bhat2, ax=ax, palette=palette)

draw_hypothesis_box(
    "sbm", 1, 0.15, yskip=0.09, ax=ax, subscript=True, xpad=0.3, ypad=0.0075
)


# ax.text(0.4, 0.93, r"$\hat{B}^{(L)}$", color=network_palette["Left"])
# ax.text(0.8, 0.93, r"$\hat{B}^{(R)}$", color=network_palette["Right"])
# compare_probability_row(1, 1, Bhat1, Bhat2, 0.9, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(1, 2, Bhat1, Bhat2, 0.85, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(1, 3, Bhat1, Bhat2, 0.8, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(2, 1, Bhat1, Bhat2, 0.75, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(2, 2, Bhat1, Bhat2, 0.7, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(2, 3, Bhat1, Bhat2, 0.65, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(3, 1, Bhat1, Bhat2, 0.6, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(3, 2, Bhat1, Bhat2, 0.55, cmap=cmap, palette=palette, ax=ax)
# compare_probability_row(3, 3, Bhat1, Bhat2, 0.5, cmap=cmap, palette=palette, ax=ax)


# ax.annotate(
#     "",
#     xy=(0.645, 0.48),
#     xytext=(0.5, 0.41),
#     arrowprops=dict(
#         arrowstyle="simple",
#         facecolor="black",
#     ),
# )


# y = 0.34
# texts = multicolor_text(
#     0.2,
#     y,
#     [r"$H_0$:", r"$B^{L}_{ij}$", r"$=$", r"$B^{R}_{ij}$"],
#     ["black", network_palette["Left"], "black", network_palette["Right"]],
#     fontsize="large",
#     ax=ax,
# )

# y = y - 0.1
# texts += multicolor_text(
#     0.2,
#     y,
#     [r"$H_A$:", r"$B^{L}_{ij}$", r"$\neq$", r"$B^{R}_{ij}$"],
#     ["black", network_palette["Left"], "black", network_palette["Right"]],
#     fontsize="large",
#     ax=ax,
# )

# bound_texts(
#     texts,
#     ax=ax,
#     xpad=0.03,
#     ypad=0.01,
#     facecolor="white",
#     edgecolor="lightgrey",
# )


ax.set_title("Compare estimated\nprobabilities", fontsize="medium")
ax.axis("off")

# ax.axis("off")


# ax = merge_axes(fig, axs, rows=None, cols=3)
ax = axs[0, 3]
uncorrected_pvalues = np.array([[0.5, 0.1, 0.22], [0.2, 0.01, 0.86], [0.43, 0.2, 0.6]])
top_ax = heatmap_grouped(
    uncorrected_pvalues,
    labels=[1, 2, 3],
    vmin=None,
    vmax=None,
    ax=ax,
    palette=palette,
    cmap="RdBu",
    center=1,
)
top_ax.set_title("Combine p-values\nfor overall test", fontsize="medium")

ax = axs[1, 3]

# generic curve that we will use for everything
# lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
# tan = np.tan(lx)
# curve = np.hstack((tan[::-1], tan))
# x0 = 0.5
# width = 0.5
# x = np.linspace(x0 - width, x0 + width, 1000)
# ax.plot(x, curve, c="k")
# ax.set(xlim)


# REF: https://stackoverflow.com/questions/50039667/matplotlib-scale-text-curly-brackets

x = 0.3
y = 1.0

trans = mtrans.Affine2D().rotate_deg_around(x, y, -90) + ax.transData
fp = FontProperties(weight="ultralight", stretch="ultra-condensed")
tp = TextPath((x, y), "$\}$", size=0.8, prop=fp, usetex=True)
pp = PathPatch(tp, lw=0, fc="k", transform=trans)
ax.add_artist(pp)
ax.set(xlim=(0, 1), ylim=(0, 1))

# ax.text(0.5, 0.5, 'p', fontsize='large')

# ax.text(0.5, 0.5, "}", rotation=-90, fontsize=100, ha='center', va='center')

# ax.plot([0, 0.4], [0.9, 0.7], color="black")
# ax.plot([0, 0.4], [0.5, 0.7], color="black")
# ax.set(xlim=(0, 1), ylim=(0.18, 1))
# ax.text(0.42, 0.7, r"$p = ...$", va="center", ha="left")

# ax.annotate(
#     "",
#     xy=(0.64, 0.68),
#     xytext=(0.5, 0.41),
#     arrowprops=dict(
#         arrowstyle="simple",
#         facecolor="black",
#     ),
# )

# y = 0.34

# y = 0.4
# texts = multicolor_text(
#     0.2,
#     y,
#     [r"$H_0$:", r"$B^{L}$", r"$=$", r"$B^{R}$"],
#     ["black", network_palette["Left"], "black", network_palette["Right"]],
#     fontsize="large",
#     ax=ax,
# )

# texts += multicolor_text(
#     0.2,
#     y - 0.15,
#     [r"$H_A$:", r"$B^{L}$", r"$\neq$", r"$B^{R}$"],
#     ["black", network_palette["Left"], "black", network_palette["Right"]],
#     fontsize="large",
#     ax=ax,
# )
# bound_texts(
#     texts,
#     ax=ax,
#     xpad=0.03,
#     ypad=0.01,
#     facecolor="white",
#     edgecolor="lightgrey",
# )

draw_hypothesis_box("sbm", 0.1, 0.4, yskip=0.15, ax=ax)

ax.axis("off")

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
print(pvalue)

#%%

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


#%% [markdown]

# ```{glue:figure} fig:sbm_unmatched_test-group_counts
# :name: "fig:sbm_unmatched_test-group_counts"

# The number of neurons in each group in each hemisphere. Note the similarity between
# the hemispheres.
# ```

#%%

fig, axs = plot_stochastic_block_probabilities(misc, network_palette)

gluefig("sbm_uncorrected", fig)

# need to save this for later for setting colorbar the same on other plot
pvalue_vmin = np.log10(np.nanmin(misc["uncorrected_pvalues"].values))
glue("pvalue_vmin", pvalue_vmin)

#%%

plot_pvalues(
    misc,
    pvalue_vmin,
    annot_missing=True,
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
    null_odds = misc["null_ratio"]
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


def plot_significant_probabilities(misc):
    B1 = misc["probabilities1"]
    B2 = misc["probabilities2"]
    null_odds = misc["null_ratio"]
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


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
