#%% [markdown]
# # Removing the Kenyon cells
#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, get_environment_variables
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import (
    SmartSVG,
    set_theme,
    svg_to_pdf,
)
from pkg.stats import erdos_renyi_test, stochastic_block_test
from pkg.utils import get_toy_palette, remove_group, sample_toy_networks
from scipy.interpolate import interp1d
from svgutils.compose import Figure, Panel, Text
from tqdm import tqdm

from giskard.plot import merge_axes, soft_axis_off

_, _, DISPLAY_FIGS = get_environment_variables()


FILENAME = "kc_thresholding"

FIG_PATH = FIG_PATH / FILENAME


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)


#%%

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()
neutral_color = sns.color_palette("Set2")[2]

left_adj, left_nodes = load_unmatched(side="left", weights=True)
right_adj, right_nodes = load_unmatched(side="right", weights=True)

GROUP_KEY = "celltype_discrete"

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values

#%%
og_sbm_stat, og_sbm_pval, og_sbm_misc = stochastic_block_test(
    left_adj, right_adj, left_labels, right_labels
)

#%%
og_asbm_stat, og_asbm_pval, og_asbm_misc = stochastic_block_test(
    left_adj, right_adj, left_labels, right_labels, density_adjustment=True
)
density_adjustment_factor = og_asbm_misc["null_ratio"]

# #%%

# from pkg.stats import compute_density_adjustment, compute_density

# density_adjustment_factor = compute_density_adjustment(left_adj, right_adj)

# #%%
# d_left = compute_density(left_adj)
# d_right = compute_density(right_adj)
# d_ratio = d_left / d_right
# d_ratio

# (d_left / (1 - d_left)) / (d_right / (1 - d_right))

#%%
# Zero out non-kenyon cell edges
left_nodes["inds"] = range(len(left_nodes))
non_kc_left_inds = left_nodes[left_nodes[GROUP_KEY] != "KCs"]["inds"].values
kc_left_inds = left_nodes[left_nodes[GROUP_KEY] == "KCs"]["inds"].values

right_nodes["inds"] = range(len(right_nodes))
non_kc_right_inds = right_nodes[right_nodes[GROUP_KEY] != "KCs"]["inds"].values
kc_right_inds = right_nodes[right_nodes[GROUP_KEY] == "KCs"]["inds"].values

left_adj[np.ix_(non_kc_left_inds, non_kc_left_inds)] = 0
right_adj[np.ix_(non_kc_right_inds, non_kc_right_inds)] = 0

from graspologic.plot import adjplot

adjplot(
    left_adj, group=left_nodes[GROUP_KEY], palette=node_palette, plot_type="scattermap"
)

adjplot(
    right_adj,
    group=right_nodes[GROUP_KEY],
    palette=node_palette,
    plot_type="scattermap",
)

#%%


def get_subgraph_parameters(A, inds):
    range_inds = np.arange(A.shape[0])
    not_inds = np.setdiff1d(range_inds, inds)

    n_edges = 0
    n_possible = 0
    group_to_non = A[inds][:, not_inds]
    n_edges += np.count_nonzero(group_to_non)
    n_possible += group_to_non.size

    non_to_group = A[not_inds][:, inds]
    n_edges += np.count_nonzero(non_to_group)
    n_possible += non_to_group.size

    group_to_group = A[inds][:, inds]
    n_edges += np.count_nonzero(group_to_group)
    # don't count loops
    n_possible += group_to_group.size - group_to_group.shape[0]

    return n_edges, n_possible


from pkg.stats import binom_2samp


def subset_erdos_renyi_test(A, B, inds_A, inds_B):
    n_edges_A, n_possible_A = get_subgraph_parameters(A, inds_A)
    n_edges_B, n_possible_B = get_subgraph_parameters(B, inds_B)

    stat, pvalue = binom_2samp(n_edges_A, n_possible_A, n_edges_B, n_possible_B)
    return stat, pvalue, {}


#%%
# dummy test just to get the subgraphs of interest
stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    left_labels,
    right_labels,
)
#%%
pvals = misc["uncorrected_pvalues"]
row_ilocs, col_ilocs = np.nonzero(pvals.notna().values)
subgraph_pairs = list(zip(pvals.index[row_ilocs], pvals.columns[col_ilocs]))

#%%

d_key = "Density"
gc_key = "Group connection"
dagc_key = "Density-adjusted\ngroup connection"


def binarize(A, threshold=None):
    # threshold is the smallest that is kept

    B = A.copy()

    if threshold is not None:
        B[B < threshold] = 0

    return B


rows = []
subgraph_rows = []
thresholds = np.arange(1, 10)

for threshold in tqdm(thresholds):
    left_adj_thresh = binarize(left_adj, threshold=threshold)
    right_adj_thresh = binarize(right_adj, threshold=threshold)

    p_edges_removed = 1 - (
        np.count_nonzero(left_adj_thresh) + np.count_nonzero(right_adj_thresh)
    ) / (np.count_nonzero(left_adj) + np.count_nonzero(right_adj))

    stat, pvalue, misc = subset_erdos_renyi_test(
        left_adj_thresh, right_adj_thresh, kc_left_inds, kc_right_inds
    )

    row = {
        "threshold": threshold,
        "stat": stat,
        "pvalue": pvalue,
        "method": d_key,
        "p_edges_removed": p_edges_removed,
    }
    rows.append(row)

    for adjusted in [False, density_adjustment_factor]:
        if adjusted:
            method = dagc_key
        else:
            method = gc_key
        stat, pvalue, misc = stochastic_block_test(
            left_adj_thresh,
            right_adj_thresh,
            left_labels,
            right_labels,
            density_adjustment=adjusted,
        )
        row = {
            "threshold": threshold,
            "adjusted": adjusted,
            "stat": stat,
            "pvalue": pvalue,
            "method": method,
            "p_edges_removed": p_edges_removed,
        }
        rows.append(row)

        uncorrected_pvalues = misc["uncorrected_pvalues"]
        for subgraph_pair in subgraph_pairs:
            pvalue = uncorrected_pvalues.loc[subgraph_pair[0], subgraph_pair[1]]
            row = {
                "threshold": threshold,
                "adjusted": adjusted,
                "pvalue": pvalue,
                "source": subgraph_pair[0],
                "target": subgraph_pair[1],
                "method": method,
                "p_edges_removed": p_edges_removed,
            }
            subgraph_rows.append(row)

integer_results = pd.DataFrame(rows)
integer_subgraph_results = pd.DataFrame(subgraph_rows)

#%%


def add_alpha_line(ax):
    ax.axhline(0.05, color="black", linestyle=":", zorder=-1)
    ax.annotate(
        r"0.05",
        (ax.get_xlim()[0], 0.05),
        xytext=(-45, -15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="black"),
        clip_on=False,
        ha="right",
    )


#%%


def plot_thresholding_pvalues(
    results, weight, figsize=(8, 6), no_reject_x=None, reject_x=None, shade=True
):
    set_theme(font_scale=1.25)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = sns.color_palette("tab20")
    palette = dict(zip([gc_key, dagc_key, d_key], [colors[0], colors[1], colors[12]]))

    sns.scatterplot(
        data=results,
        x="p_edges_removed",
        y="pvalue",
        hue="method",
        palette=palette,
        ax=ax,
        legend=True,
    )
    sns.lineplot(
        data=results,
        x="p_edges_removed",
        y="pvalue",
        hue="method",
        palette=palette,
        ax=ax,
        legend=False,
    )

    ax.set(
        yscale="log",
        ylabel="p-value",
        xlabel="Edges removed",
        yticks=np.geomspace(1, 1e-20, 5),
    )

    # just pick any method because these are the same for each
    single_results = results[results["method"] == "Density"]
    x = single_results["p_edges_removed"]
    y = single_results["threshold"]

    ax.set_xlim((x.min(), x.max()))
    ax.tick_params(axis="both", length=5)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["0%", "25%", "50%", "75%"])

    # basically fitting splines to interpolate linearly between points we checked
    prop_to_thresh = interp1d(
        x=x, y=y, kind="slinear", bounds_error=False, fill_value=(0, 1)
    )
    thresh_to_prop = interp1d(
        x=y, y=x, kind="slinear", bounds_error=False, fill_value=(0, 1)
    )

    ax2 = ax.secondary_xaxis(-0.2, functions=(prop_to_thresh, thresh_to_prop))

    if weight == "input_proportion":
        ax2.set_xticks([0.005, 0.01, 0.015, 0.02])
        ax2.set_xticklabels(["0.5%", "1%", "1.5%", "2%"])
        ax2.set_xlabel("Weight threshold (input percentage)")
    elif weight == "synapse_count":
        ax2.set_xlabel("Weight threshold (synapse count)")
    ax2.tick_params(axis="both", length=5)

    add_alpha_line(ax)

    sns.move_legend(
        ax,
        "lower left",
        title="Test",
        frameon=True,
        fontsize="small",
        ncol=1,
        labelspacing=0.3,
    )

    # shading
    ax.autoscale(False)

    if shade:
        if no_reject_x is None:
            no_reject_x = ax.get_xlim()[1]
        ax.fill_between(
            (ax.get_xlim()[0], no_reject_x),
            y1=0.05,
            y2=ax.get_ylim()[0],
            color="darkred",
            alpha=0.05,
        )

        if reject_x is None:
            reject_x = np.mean(ax.get_xlim())

        y = np.mean(np.sqrt(np.product(ax.get_ylim())))
        ax.text(
            reject_x,
            y,
            "Reject\nsymmetry",
            ha="center",
            va="center",
            color="darkred",
        )

    return fig, ax


#%%
fig, ax = plot_thresholding_pvalues(integer_results, "synapse_count", shade=False)
gluefig("synapse_threshold_pvalues_legend", fig)
ax.get_legend().remove()
gluefig("synapse_threshold_pvalues", fig)


# %%

#%%
### EDGE WEIGHTS AS INPUT PROPORTIONS
#%%
left_input = (left_nodes["axon_input"] + left_nodes["dendrite_input"]).values
left_input[left_input == 0] = 1
left_adj_input_norm = left_adj / left_input[None, :]

right_input = (right_nodes["axon_input"] + right_nodes["dendrite_input"]).values
right_input[right_input == 0] = 1
right_adj_input_norm = right_adj / right_input[None, :]

#%%

rows = []
input_subgraph_rows = []
thresholds = np.linspace(0, 0.03, 31)
for threshold in tqdm(thresholds):
    left_adj_thresh = binarize(left_adj_input_norm, threshold=threshold)
    right_adj_thresh = binarize(right_adj_input_norm, threshold=threshold)

    p_edges_removed = 1 - (
        np.count_nonzero(left_adj_thresh) + np.count_nonzero(right_adj_thresh)
    ) / (np.count_nonzero(left_adj) + np.count_nonzero(right_adj))

    stat, pvalue, misc = subset_erdos_renyi_test(
        left_adj_thresh, right_adj_thresh, kc_left_inds, kc_right_inds
    )

    row = {
        "threshold": threshold,
        "stat": stat,
        "pvalue": pvalue,
        "method": d_key,
        "p_edges_removed": p_edges_removed,
    }
    rows.append(row)

    for adjusted in [False, True]:
        if adjusted:
            method = dagc_key
        else:
            method = gc_key
        stat, pvalue, misc = stochastic_block_test(
            left_adj_thresh,
            right_adj_thresh,
            left_labels,
            right_labels,
            density_adjustment=adjusted,
        )
        row = {
            "threshold": threshold,
            "adjusted": adjusted,
            "stat": stat,
            "pvalue": pvalue,
            "method": method,
            "p_edges_removed": p_edges_removed,
        }
        rows.append(row)

        uncorrected_pvalues = misc["uncorrected_pvalues"]
        for subgraph_pair in subgraph_pairs:
            pvalue = uncorrected_pvalues.loc[subgraph_pair[0], subgraph_pair[1]]
            row = {
                "threshold": threshold,
                "adjusted": adjusted,
                "pvalue": pvalue,
                "source": subgraph_pair[0],
                "target": subgraph_pair[1],
                "method": method,
                "p_edges_removed": p_edges_removed,
            }
            subgraph_rows.append(row)

input_results = pd.DataFrame(rows)
input_subgraph_results = pd.DataFrame(subgraph_rows)

#%%


x = input_results[input_results["method"] == "Density"].iloc[17]["p_edges_removed"]
x_threshold = input_results[input_results["method"] == "Density"].iloc[17]["threshold"]
# no_reject_x=x, reject_x=np.mean((0, x))
fig, ax = plot_thresholding_pvalues(input_results, "input_proportion", shade=False)

# ax.axvline(
#     x,
#     ax.get_ylim()[0],
#     0.95,
#     color="black",
#     linestyle="--",
#     zorder=0,
# )
# ax.text(
#     x + 0.005,
#     np.mean(np.sqrt(np.product(ax.get_ylim()))),
#     r"$\rightarrow$"
#     + "\n\n\n  All tests fail to\n  reject symmetry\n\n\n"
#     + r"$\rightarrow$",
#     ha="left",
#     va="center",
# )

gluefig("input_threshold_pvalues_legend", fig)
ax.get_legend().remove()
gluefig("input_threshold_pvalues", fig)

#%%

#%%


fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)

interesting_subgraph_pairs = [
    ("KCs", "CNs"),
    ("KCs", "KCs"),
    ("KCs", "MB-FBNs"),
    ("MBONs", "KCs"),
]

bonferonni_threshold = 0.05 / og_sbm_misc["n_tests"]


def add_alpha_line(ax, threshold, label=True):
    ax.axhline(threshold, color="black", linestyle=":", zorder=-1)
    if label:
        ax.annotate(
            r"$\alpha^*$",
            (ax.get_xlim()[0], threshold),
            xytext=(-45, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="black"),
            clip_on=False,
            ha="right",
        )


colors = sns.color_palette("tab20")
palette = dict(zip([gc_key, dagc_key, d_key], [colors[0], colors[1], colors[12]]))

for i, subgraph_pair in enumerate(interesting_subgraph_pairs):
    # thresholding_results = integer_subgraph_results.query(
    #     "('source' == @subgraph_pair[0]) & ('target' == @subgraph_pair[1])"
    # )
    thresholding_results = integer_subgraph_results[
        (integer_subgraph_results["source"] == subgraph_pair[0])
        & (integer_subgraph_results["target"] == subgraph_pair[1])
    ]

    ax = axs[0, i]
    sns.lineplot(
        data=thresholding_results,
        x="p_edges_removed",
        y="pvalue",
        hue="method",
        ax=ax,
        palette=palette,
        legend=False,
    )
    sns.scatterplot(
        data=thresholding_results,
        x="p_edges_removed",
        y="pvalue",
        hue="method",
        ax=ax,
        palette=palette,
        legend=True,
    )
    if i:
        ax.get_legend().remove()
    else:
        sns.move_legend(
            ax, "lower right", frameon=True, title="Test", fontsize="x-small"
        )
    ax.set_title(subgraph_pair[0] + r"$\rightarrow$" + subgraph_pair[1])
    ax.axhline(bonferonni_threshold, color="black", linestyle="--", zorder=0)
    add_alpha_line(ax, bonferonni_threshold, label=i == 0)


for i, subgraph_pair in enumerate(interesting_subgraph_pairs):
    # thresholding_results = integer_subgraph_results.query(
    #     "('source' == @subgraph_pair[0]) & ('target' == @subgraph_pair[1])"
    # )
    thresholding_results = input_subgraph_results[
        (input_subgraph_results["source"] == subgraph_pair[0])
        & (input_subgraph_results["target"] == subgraph_pair[1])
    ]

    ax = axs[1, i]
    sns.lineplot(
        data=thresholding_results,
        x="p_edges_removed",
        y="pvalue",
        hue="method",
        ax=ax,
        palette=palette,
    )
    sns.scatterplot(
        data=thresholding_results,
        x="p_edges_removed",
        y="pvalue",
        hue="method",
        ax=ax,
        palette=palette,
    )
    ax.get_legend().remove()
    ax.set_title(subgraph_pair[0] + r"$\rightarrow$" + subgraph_pair[1])
    ax.axhline(bonferonni_threshold, color="black", linestyle="--", zorder=0)
    add_alpha_line(ax, bonferonni_threshold, label=i == 0)

    ax.set(
        ylabel="p-value",
        xlabel="Edges removed",
    )

    ax.tick_params(axis="both", length=5)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["0%", "25%", "50%", "75%"])

ax.set_yscale("log")


fig.text(
    -0.02,
    0.72,
    "Synapse\nthresholding",
    va="center",
    ha="center",
    fontsize="large",
)
fig.text(
    -0.02,
    0.32,
    "Input\nthresholding",
    va="center",
    ha="center",
    fontsize="large",
)

gluefig("kc_thresholding_subgraphs", fig)

# %%

#%%
fontsize = 9

synapse_pvalues = SmartSVG(FIG_PATH / "synapse_threshold_pvalues_legend.svg")
synapse_pvalues.set_width(200)
synapse_pvalues.move(10, 15)
synapse_pvalues_panel = Panel(
    synapse_pvalues,
    Text("A) Synapse thresholding p-values", 5, 10, size=fontsize, weight="bold"),
)

input_pvalues = SmartSVG(FIG_PATH / "input_threshold_pvalues_legend.svg")
input_pvalues.set_width(200)
input_pvalues.move(10, 15)
input_pvalues_panel = Panel(
    input_pvalues,
    Text("B) Input thresholding p-values", 5, 10, size=fontsize, weight="bold"),
)
input_pvalues_panel.move(synapse_pvalues.width * 0.85, 0)

fig = Figure(
    synapse_pvalues.width * 2 * 0.88,
    (synapse_pvalues.height) * 0.92,
    synapse_pvalues_panel,
    input_pvalues_panel,
)
fig.save(FIG_PATH / "kc_thresholding_composite.svg")

svg_to_pdf(
    FIG_PATH / "kc_thresholding_composite.svg",
    FIG_PATH / "kc_thresholding_composite.pdf",
)

fig
