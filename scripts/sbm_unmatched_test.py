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
from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import rotate_labels
from matplotlib.transforms import Bbox
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.perturb import remove_edges
from pkg.plot import set_theme
from pkg.stats import stochastic_block_test
from seaborn.utils import relative_luminance


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
#
# $$ A \sim SBM(B, \tau)$$
#
# We say that for all $(i,j), i \neq j$, with $i$ and $j$ both running
# from $1 ... n$ the probability of edge $(i,j)$ occuring is:
#
# $$ P[A_{ij} = 1] = P_{ij} = B_{\tau_i, \tau_j} $$
#
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
#
# ````{admonition} Math
# We are interested in testing:
#
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
#
# ```{admonition} Math
# The hypothesis test above can be decomposed into $K^2$ indpendent hypotheses.
# $B^{(L)}$
# and $B^{(R)}$ are both $K \times K$ matrices, where each element $b_{kl}$ represents
# the probability of a connection from a neuron in group $k$ to one in group $l$. We
# also know that group $k$ for the left network corresponds with group $k$ for the
# right. In other words, the *groups* are matched. Thus, we are interested in testing,
# for $k, l$ both running from $1...K$:
#
# $$ H_0: B_{kl}^{(L)} = B_{kl}^{(R)},
# \quad H_A: B_{kl}^{(L)} \neq B_{kl}^{(R)}$$
#
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

stat, pvalue, misc = stochastic_block_test(
    left_adj, right_adj, labels1=left_labels, labels2=right_labels, method="fisher"
)
glue("uncorrected_pvalue", pvalue)
n_tests = misc["n_tests"]
glue("n_tests", n_tests)

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

#%% [markdown]

# ```{glue:figure} fig:sbm_unmatched_test-group_counts
# :name: "fig:sbm_unmatched_test-group_counts"

# The number of neurons in each group in each hemisphere. Note the similarity between
# the hemispheres.
# ```

#%%


def plot_stochastic_block_test(misc, pvalue_vmin=None):
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
    width_ratios = [0.5, pad + 0.8, 10, pad - 0.4, 10, pad + 0.9, 10, 0.5]
    set_theme(font_scale=1.25)
    fig, axs = plt.subplots(
        1,
        len(width_ratios),
        figsize=(30, 10),
        gridspec_kw=dict(
            width_ratios=width_ratios,
        ),
    )
    left_col = 2
    right_col = 4
    pvalue_col = 6

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

    # handle the colorbars
    # NOTE: did it this way cause the other options weren't playing nice with auto
    # constrain
    # layouts.

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

    ax = axs[0]
    shrink_axis(ax, scale=0.5)
    _ = fig.colorbar(
        im.get_children()[0],
        cax=ax,
        fraction=1,
        shrink=1,
        ticklocation="left",
    )

    # plot p-values
    ax = axs[pvalue_col]

    annot = np.full((K, K), "")
    annot[(B1.values == 0) & (B2.values == 0)] = "B"
    annot[(B1.values == 0) & (B2.values != 0)] = "L"
    annot[(B1.values != 0) & (B2.values == 0)] = "R"
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
    ax.set(ylabel="", xlabel="Target group")
    ax.set(xticks=np.arange(K) + 0.5, xticklabels=index)
    ax.set_title(r"$log_{10}($p-value$)$", fontsize="xx-large")

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
    ax = axs[7]
    shrink_axis(ax, scale=0.5)
    _ = fig.colorbar(
        im.get_children()[0],
        cax=ax,
        fraction=1,
        shrink=1,
        ticklocation="right",
    )

    fig.text(0.11, 0.85, "A)", fontweight="bold", fontsize=50)
    fig.text(0.63, 0.85, "B)", fontweight="bold", fontsize=50)

    # remove dummy axes
    for i in range(len(width_ratios)):
        if not axs[i].has_data():
            axs[i].set_visible(False)

    return fig, axs


fig, axs = plot_stochastic_block_test(misc)
gluefig("sbm_uncorrected", fig)

# need to save this for later for setting colorbar the same on other plot
pvalue_vmin = np.log10(np.nanmin(misc["uncorrected_pvalues"].values))

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
        pair = source + r"$\rightarrow$" + target
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.pointplot(
        data=sig_data,
        y="p",
        x="pair",
        ax=ax,
        hue="side",
        dodge=True,
        join=False,
        palette=network_palette,
    )

    ax.get_legend().set_title("Side")
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


#%% [markdown]
# These observations are consistent with the idea that perhaps the probabilities
# on the right are a scaled up version of those on the right, for some global scaling.
# We can frame this question as a new null hypothesis:
#
# ````{admonition} Math
# With variables defined as in Equation {eq}`sbm_unmatched_null`, we can write our new
# null hypothesis as:
# ```{math}
# :label: sbm_unmatched_null_adjusted
# H_0: B^{(L)} = c B^{(R)}, \quad H_A: B^{(L)} \neq c B^{(R)}
# ```
# where $c$ is the ratio of the densities, $c = \frac{p^{(L)}}{p^{(R)}}$.
# ````

#%% [markdown]
# ### Correcting by subsampling edges for one network
# One naive (though quite intuitive) approach to adjust our test for a difference in
# density is to simply make the densities of the two networks the same and then rerun
# our
# test. To do so, we calculated the number of edge removals (from the right hemisphere)
# required to set the network densities roughly the same. We then randomly removed
# that many edges from the right hemisphere network and
# then re-ran the SBM test procedure above. We repeated this procedure
# {glue:text}`sbm_unmatched_test-n_resamples` times, resulting in a p-value for each
# subsampling of the right network.
#
# The distribution of p-values from this process is
# shown in {numref}`Figure {number} <fig:sbm_unmatched_test-pvalues_corrected>`. Whereas
# the p-value for the original null hypothesis was
# {glue:text}`sbm_unmatched_test-uncorrected_pvalue:0.2e`, we see now that the p-values
# from our subsampled, density-adjusted test are around 0.8, indicating insufficient
# evidence to reject our density-adjusted null hypothesis of bilateral symmetry
# (Equation {eq}`sbm_unmatched_null_adjusted`).
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
    stat, pvalue, misc = stochastic_block_test(
        left_adj,
        subsampled_right_adj,
        labels1=left_labels,
        labels2=right_labels,
        method="fisher",
    )
    rows.append({"stat": stat, "pvalue": pvalue, "misc": misc, "resample": i})

resample_results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(data=resample_results, x="pvalue", ax=ax)
ax.set(xlabel="p-value", ylabel="", yticks=[])
ax.spines["left"].set_visible(False)

mean_resample_pvalue = np.mean(resample_results["pvalue"])
median_resample_pvalue = np.median(resample_results["pvalue"])

gluefig("pvalues_corrected", fig)

#%% [markdown]
# ```{glue:figure} fig:sbm_unmatched_test-pvalues_corrected
# :name: "fig:sbm_unmatched_test-pvalues_corrected"
#
# Histogram of p-values after a correction for network density. For the observed
# networks
# the left hemisphere has a density of
# {glue:text}`sbm_unmatched_test-density_left:0.4f`, and the right
# hemisphere has
# a density of
# {glue:text}`sbm_unmatched_test-density_right:0.4f`. Here, we randomly removed exactly
# {glue:text}`sbm_unmatched_test-n_remove`
# edges from the right hemisphere network, which makes the density of the right network
# match that of the left hemisphere network. Then, we re-ran the stochastic block model
# testing
# procedure from {numref}`Figure {number} <fig:sbm_unmatched_test-sbm_uncorrected>`.
# This entire process
# was repeated {glue:text}`sbm_unmatched_test-n_resamples` times. The histogram above
# shows the
# distribution
# of p-values for the overall test. Note that the p-values are no longer small,
# indicating
# that with this density correction, we now failed to reject our null hypothesis of
# bilateral symmetry under the stochastic block model.
# ```

#%% [markdown]
# ### An analytic approach to correcting for differences in density
# Instead of randomly resetting the density of the right hemisphere network, we can
# actually modify the hypothesis we are testing for each element of the $\hat{B}$
# matrices to include this adjustment by some constant scale, $c$.
#
# ```{admonition} Math
# Fisher's exact test (used
# above to compare each element of the $\hat{B}$ matrices) tests the null hypotheses:
#
# $$H_0: B_{kl}^{(L)} = B_{kl}^{(R)}, \quad H_A: B_{kl}^{(L)} \neq B_{kl}^{(R)}$$
#
# for each $(k, l)$ pair, where $k$ and $l$ are the indices of the source and target
# groups, respectively.
#
# Instead, we can use a test of:
#
# $$H_0: B_{kl}^{(L)} = c B_{kl}^{(R)}, \quad H_A: B_{kl}^{(L)} \neq c B_{kl}^{(R)}$$
#
# In our case, $c$ is a constant that we fit to the entire right hemisphere network to
# set its density equal to the left, $c = \frac{p^{(L)}}{p_{(R)}}$
#
# A test for the adjusted null hypothesis above is given by using
# [Fisher's noncentral hypergeometric distribution
# ](https://en.wikipedia.org/wiki/Fisher%27s_noncentral_hypergeometric_distribution)
# and applying a procedure much like that of the traditional Fisher's exact test.
# ```
#
# More information about this test can be found in [](nhypergeom_sims).
#%%
null_odds = density_left / density_right
stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    null_odds=null_odds,
)
glue("corrected_pvalue", pvalue)

#%%
fig, axs = plot_stochastic_block_test(misc, pvalue_vmin=pvalue_vmin)
gluefig("sbm_corrected", fig)

#%% [markdown]
# {numref}`Figure {number} <fig:sbm_unmatched_test-sbm_corrected>` shows the results
# of running the analytic version of the density-adjusted test based on Fisher's
# noncentral hypergeometric distribution. Note that now only two group-to-group
# probability comparisons are significant after Bonferroni-Holm correction, and the
# overall p-value for this test of Equation {eq}`sbm_unmatched_null_adjusted` is
# {glue:text}`sbm_unmatched_test-corrected_pvalue:0.2f`.

#%% [markdown]
# ```{glue:figure} fig:sbm_unmatched_test-sbm_corrected
# :name: "fig:sbm_unmatched_test-sbm_corrected"

# Comparison of stochastic block model fits for the left and right hemispheres after
# correcting for a difference in hemisphere density.
# **A)** The estimated group-to-group connection probabilities for the left
# and right hemispheres, after the right hemisphere probabilities were scaled by a
# density-adjusting constant, $c$. Any estimated
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
# null hypothesis that the two group connection probability matrices are the same after
# adjustment by a density-normalizing constant, $c$) of
# {glue:text}`sbm_unmatched_test-corrected_pvalue:0.2f`.
# ```

#%% [markdown]
# Taken together, these results suggest that for the unmatched networks, and using the
# known cell type labels, we reject the null hypothesis of bilateral symmetry under the
# SBM (Equation {eq}`sbm_unmatched_null`), but fail to reject the null hypothesis of
# bilateral symmetry under the SBM after a density adjustment (Equation
# {eq}`sbm_unmatched_null_adjusted`). Moreover, they highlight the insights that
# can be gained
# by considering multiple definitions of bilateral symmetry.

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
