#%%

import datetime
import time
from giskard.plot.utils import merge_axes, soft_axis_off

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import load_maggot_graph, select_nice_nodes
from pkg.io import FIG_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import SmartSVG, layout, set_theme
from pkg.stats import erdos_renyi_test, stochastic_block_test
from svgutils.compose import Figure, Panel, Text
from tqdm import tqdm


DISPLAY_FIGS = True

FILENAME = "thresholding_tests"

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme()


#%%

from pkg.data import load_unmatched
from pkg.data import load_network_palette

network_palette, NETWORK_KEY = load_network_palette()

left_adj, left_nodes = load_unmatched("left", weights=True)
right_adj, right_nodes = load_unmatched("right", weights=True)

GROUP_KEY = "simple_group"

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values

#%%
# from pkg.plot import simple_plot_neurons

# pickle_path = "bilateral-connectome/data/2021-05-24-v2/neurons.pickle"

# import pickle
# from navis import NeuronList

# with open(pickle_path, "rb") as f:
#     neuronlist = pickle.load(f)

# neutral_color = sns.color_palette("Set2")[2]

# fig = plt.figure(figsize=(5, 5), constrained_layout=True)
# gs = plt.GridSpec(1, 1, figure=fig)
# ax = fig.add_subplot(gs[0, 0], projection="3d")
# # edge weight definitions
# # 4 6
# # 7 as a pre?
# # 8 as a post?
# # 13 as pre
# neuron = NeuronList(neuronlist[13])
# name = int(neuron.id[0])
# simple_plot_neurons(
#     neuron,
#     ax=ax,
#     dist=3,
#     axes_equal=True,
#     palette={name: neutral_color},
#     use_x=True,
#     use_y=True,
#     use_z=False,
#     force_bounds=True,
#     axis_off=False,
#     linewidth=1,
# )

# #%%

# fig, axs = plt.subplots(2, 1, figsize=(4, 5))

# import networkx as nx

# g = nx.DiGraph()
# g.add_edge(0, 1, weight=4)
# pos = {0: (0.1, 0.65), 1: (0.5, 0.65)}
# ax = axs[0]
# soft_axis_off(ax)
# nx.draw_networkx(
#     g,
#     pos,
#     ax=ax,
#     arrowstyle="-|>",
#     connectionstyle="arc3,rad=-0.5",
#     width=3,
#     node_size=500,
#     with_labels=False,
# )
# ax.set(xlim=(0, 1), ylim=(0, 1))
# ax.set_ylabel("Synapse\ncount", rotation=0, ha="right")
# ax.text(0.65, 0.65, "Weight = 2", fontsize="medium")

# ax = axs[1]
# soft_axis_off(ax)
# nx.draw_networkx(
#     g,
#     pos,
#     ax=ax,
#     arrowstyle="-|>",
#     connectionstyle="arc3,rad=-0.5",
#     width=3,
#     node_size=500,
#     with_labels=False,
# )
# ax.set(xlim=(0, 1), ylim=(0, 1))
# # $\frac{2}{5}$
# ax.text(0.65, 0.65, "Weight = 2 / 5", fontsize="medium")


# def draw_input(xytext):
#     ax.annotate(
#         "",
#         (0.5, 0.65),
#         xytext=xytext,
#         textcoords="offset points",
#         arrowprops=dict(
#             arrowstyle="-|>",
#             connectionstyle="arc3",
#             facecolor="grey",
#             linewidth=3,
#             shrinkB=10,
#             edgecolor="grey"
#             # width=0.5,
#             # mutation_scale=0.5,
#         ),
#     )


# draw_input((-25, -25))
# draw_input((0, -35))
# draw_input((-35, 0))
# draw_input((0, 35))

# ax.set_ylabel("Input\nproportion", rotation=0, ha="right")

# fig.set_facecolor("w")

#%%
rng = np.random.default_rng(8888)

from pkg.utils import sample_toy_networks
import networkx as nx

A1, A2, node_data = sample_toy_networks()

node_data["labels"] = np.ones(len(node_data), dtype=int)
palette = {1: sns.color_palette("Set2")[2]}

g1 = nx.from_numpy_array(A1)
g2 = nx.from_numpy_array(A2)

pos1 = nx.kamada_kawai_layout(g1)
pos2 = nx.kamada_kawai_layout(g2)


def weight_adjacency(A, scale=6):
    A = A.copy()
    sources, targets = np.nonzero(A)
    for source, target in zip(sources, targets):
        # weight = rng.poisson(scale)
        weight = rng.uniform(1, 10)
        A[source, target] = weight
    return A


from giskard.plot import soft_axis_off


def layoutplot(
    g,
    pos,
    nodes,
    ax=None,
    figsize=(10, 10),
    weight_scale=1,
    node_alpha=1,
    node_size=300,
    palette=None,
    edge_alpha=0.4,
    edge_color="black",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    edgelist = g.edges()
    weights = np.array([g[u][v]["weight"] for u, v in edgelist])
    # weight transformations happen here, can be important

    weights *= weight_scale

    # plot the actual layout
    # nx.draw_networkx_nodes(
    #     g,
    #     pos,
    #     nodelist=nodes.index,
    #     node_color="black",
    #     node_size=node_size * 1.05,
    #     ax=ax,
    # )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes.index,
        node_color=nodes["labels"].map(palette),
        edgecolors="black",
        alpha=node_alpha,
        node_size=node_size,
        ax=ax,
    )

    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=edgelist,
        nodelist=nodes.index,
        width=weights,
        edge_vmin=-3,
        edge_vmax=9,
        edge_color=weights,
        alpha=edge_alpha,
        ax=ax,
        node_size=node_size,  # connectionstyle="arc3,rad=0.2"
    )

    soft_axis_off(ax)

    return ax


set_theme(font_scale=1.75)

fig, axs = plt.subplots(
    4,
    3,
    figsize=(12, 10),
    constrained_layout=True,
    gridspec_kw=dict(height_ratios=[0.5, 1, 0.25, 1], hspace=0, wspace=0),
)
A1 = weight_adjacency(A1)
A2 = weight_adjacency(A2)
kwargs = dict(
    palette=palette, edge_alpha=1, edge_color=(0.65, 0.65, 0.65), weight_scale=0.75
)
thresholds = [1, 4, 7]
for i in range(3):
    A1[A1 < thresholds[i]] = 0
    A2[A2 < thresholds[i]] = 0
    g1 = nx.from_numpy_array(A1)
    g2 = nx.from_numpy_array(A2)

    ax = axs[1, i]
    layoutplot(g1, pos1, node_data, ax=ax, **kwargs)
    ax = axs[3, i]
    layoutplot(g2, pos2, node_data, ax=ax, **kwargs)

from giskard.plot import merge_axes
from pkg.plot import rainbowarrow

ax = merge_axes(fig, axs, rows=0)

rainbowarrow(ax, start=(0.1, 0.5), end=(0.9, 0.5), cmap="Greys", n=1000, lw=30)
ax.set(ylim=(0.4, 0.8), xlim=(0, 1))
ax.set_title("Increasing edge weight threshold", fontsize="medium", y=0.5)
ax.axis("off")


def draw_comparison(ax):
    ax.text(
        0.48, 0.35, r"$\overset{?}{=}$", fontsize="xx-large", ha="center", va="center"
    )
    # ax.plot([0.5, 0.5], [-0.5, 1.25], clip_on=False, linewidth=2, color='darkgrey')
    ax.set(ylim=(0, 1), xlim=(0, 1))
    ax.axis("off")


ax = axs[2, 0]
draw_comparison(ax)

ax = axs[2, 1]
draw_comparison(ax)

ax = axs[2, 2]
draw_comparison(ax)

ax.annotate(
    "Rerun all\n tests",
    (0.6, 0.6),
    xytext=(45, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-|>", facecolor="black"),
    va="center",
)

# ax = merge_axes(fig, axs, rows=2)
# ax.axis("off")

axs[1, 0].set_ylabel(
    "Left",
    color=network_palette["Left"],
    size="large",
    rotation=0,
    ha="right",
    labelpad=10,
)

axs[3, 0].set_ylabel(
    "Right",
    color=network_palette["Right"],
    size="large",
    rotation=0,
    ha="right",
    labelpad=10,
)

fig.set_facecolor("w")

gluefig("thresholding_methods", fig)

# %%


def construct_weight_data(left_adj, right_adj):
    indices = np.nonzero(left_adj)
    left_weights = left_adj[indices]

    indices = np.nonzero(right_adj)
    right_weights = right_adj[indices]

    labels = np.concatenate(
        (len(left_weights) * ["Left"], len(right_weights) * ["Right"])
    )
    weights = np.concatenate((left_weights, right_weights))
    weight_data = pd.Series(data=weights, name="weights").to_frame()
    weight_data["labels"] = labels
    return weight_data


weight_data = construct_weight_data(left_adj, right_adj)
weight_data = weight_data[weight_data["weights"] < 10]


set_theme(font_scale=1.25)
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

sns.histplot(
    data=weight_data,
    x="weights",
    hue="labels",
    palette=network_palette,
    ax=ax,
    discrete=True,
    cumulative=False,
)
sns.move_legend(ax, loc="upper right", title="Hemisphere")
ax.set(xlabel="Weight (synapse count)")
ax.set(xticks=np.arange(1, 10))
ax.set_yscale("log")
# ax.set(xlim=(0, 10))

gluefig("synapse_weight_histogram", fig)

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
thresholds = np.arange(1, 10)
for threshold in tqdm(thresholds):
    left_adj_thresh = binarize(left_adj, threshold=threshold)
    right_adj_thresh = binarize(right_adj, threshold=threshold)

    p_edges_removed = 1 - (
        np.count_nonzero(left_adj_thresh) + np.count_nonzero(right_adj_thresh)
    ) / (np.count_nonzero(left_adj) + np.count_nonzero(right_adj))

    stat, pvalue, misc = erdos_renyi_test(left_adj_thresh, right_adj_thresh)
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

integer_results = pd.DataFrame(rows)


#%%
set_theme(font_scale=1)


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


colors = sns.color_palette("tab20")
palette = dict(zip([gc_key, dagc_key, d_key], colors))

fig, ax = plt.subplots(1, 1, figsize=(7, 6))


sns.scatterplot(
    data=integer_results,
    x="threshold",
    y="pvalue",
    hue="method",
    palette=palette,
    ax=ax,
)
sns.lineplot(
    data=integer_results,
    x="threshold",
    y="pvalue",
    hue="method",
    palette=palette,
    ax=ax,
    legend=False,
)
ax.set(yscale="log", ylabel="p-value", xlabel="Edge weight (# synapses) threshold")
sns.move_legend(ax, "lower right", title="Test", frameon=True, fontsize="small")
ax.set(xticks=thresholds)
add_alpha_line(ax)

gluefig("integer_threshold_pvalues", fig)


#%%
fig, ax = plt.subplots(1, 1, figsize=(7, 6))


sns.scatterplot(
    data=integer_results,
    x="p_edges_removed",
    y="pvalue",
    hue="method",
    palette=palette,
    ax=ax,
    legend=False,
)
sns.lineplot(
    data=integer_results,
    x="p_edges_removed",
    y="pvalue",
    hue="method",
    palette=palette,
    ax=ax,
    legend=False,
)
ax.set(
    yscale="log", ylabel="p-value", xlabel="Proportion of edges removed (by # synapses)"
)
add_alpha_line(ax)
gluefig("integer_threshold_pvalues_p_removed", fig)

#%%
left_input = (left_nodes["axon_input"] + left_nodes["dendrite_input"]).values
left_input[left_input == 0] = 1
left_adj_input_norm = left_adj / left_input[None, :]

right_input = (right_nodes["axon_input"] + right_nodes["dendrite_input"]).values
right_input[right_input == 0] = 1
right_adj_input_norm = right_adj / right_input[None, :]

#%%
set_theme(font_scale=1.25)
weight_data = construct_weight_data(left_adj_input_norm, right_adj_input_norm)
median = np.median(weight_data["weights"])
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
sns.histplot(
    data=weight_data,
    x="weights",
    hue="labels",
    palette=network_palette,
    ax=ax,
    discrete=False,
    cumulative=False,
    common_norm=True,
    log_scale=True,
    stat="count",
    bins=50,
    kde=True,
)
sns.move_legend(ax, loc="upper right", title="Hemisphere")
ax.set(xlabel="Weight (input proportion)")
neutral_color = sns.color_palette("Set2")[2]
ax.axvline(median, color=neutral_color, linestyle="--", linewidth=4, alpha=1)
ax.text(
    median - 0.001,
    ax.get_ylim()[1],
    "Median",
    ha="right",
    va="top",
    color=neutral_color,
)
gluefig("input_proportion_histogram", fig)
#%%

from pkg.utils import remove_group

(
    left_adj_input_norm_sub,
    right_adj_input_norm_sub,
    left_nodes_sub,
    right_nodes_sub,
) = remove_group(
    left_adj_input_norm, right_adj_input_norm, left_nodes, right_nodes, "KCs"
)


weight_data = construct_weight_data(left_adj_input_norm_sub, right_adj_input_norm_sub)
median = np.median(weight_data["weights"])
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(
    data=weight_data,
    x="weights",
    hue="labels",
    palette=network_palette,
    ax=ax,
    discrete=False,
    cumulative=False,
    common_norm=True,
    log_scale=True,
    stat="count",
    bins=50,
    kde=True,
)
sns.move_legend(ax, loc="upper right", title="Hemisphere")
ax.set(xlabel="Weight (input proportion)")
neutral_color = sns.color_palette("Set2")[2]
ax.axvline(median, color=neutral_color, linestyle="--", linewidth=4, alpha=1)
ax.text(
    median - 0.001,
    ax.get_ylim()[1],
    "Median",
    ha="right",
    va="top",
    color=neutral_color,
)
ax.set_title("KC-")
# gluefig("edge_weight_dist_input_proportion", fig)


#%%


rows = []
thresholds = np.linspace(0, 0.03, 31)
for threshold in tqdm(thresholds):
    left_adj_thresh = binarize(left_adj_input_norm, threshold=threshold)
    right_adj_thresh = binarize(right_adj_input_norm, threshold=threshold)

    p_edges_removed = 1 - (
        np.count_nonzero(left_adj_thresh) + np.count_nonzero(right_adj_thresh)
    ) / (np.count_nonzero(left_adj) + np.count_nonzero(right_adj))

    stat, pvalue, misc = erdos_renyi_test(left_adj_thresh, right_adj_thresh)
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
input_results = pd.DataFrame(rows)
input_results

# %%
set_theme(font_scale=1)

fig, ax = plt.subplots(1, 1, figsize=(7, 6))


sns.scatterplot(
    data=input_results,
    x="threshold",
    y="pvalue",
    hue="method",
    palette=palette,
    ax=ax,
    legend=False,
)
sns.lineplot(
    data=input_results,
    x="threshold",
    y="pvalue",
    hue="method",
    palette=palette,
    ax=ax,
    legend=False,
)
ax.set(yscale="log", ylabel="p-value", xlabel="Edge weight (relative input) threshold")
ax.set(xticks=thresholds, yticks=np.geomspace(1, 1e-21, 8))
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
add_alpha_line(ax)

gluefig("input_threshold_pvalues", fig)

#%%
from scipy.interpolate import interp1d

set_theme(font_scale=1)


def plot_thresholding_pvalues(
    results, weight, figsize=(8, 6), no_reject_x=None, reject_x=None
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = sns.color_palette("tab20")
    palette = dict(zip([gc_key, dagc_key, d_key], colors))

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
        yticks=np.geomspace(1, 1e-21, 8),
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
    # ax2 = ax.secondary_xaxis(-0.2, functions=(thresh_to_prop, prop_to_thresh))

    if weight == "input_proportion":
        ax2.set_xticks([0.005, 0.01, 0.015, 0.02])
        ax2.set_xticklabels(["0.5%", "1%", "1.5%", "2%"])
        ax2.set_xlabel("Weight threshold (input proportion)")
    elif weight == "synapse_count":
        # ax2.set_xticks([2.0, 4.0, 6.0, 8.0])
        # ax2.set_xticklabels(["0.5%", "1%", "1.5%", "2%"])
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
        "Reject symmetry",
        ha="center",
        va="center",
        color="darkred",
    )

    return fig, ax


#%%
fig, ax = plot_thresholding_pvalues(integer_results, "synapse_count")
gluefig("synapse_threshold_pvalues_legend", fig)
ax.get_legend().remove()
gluefig("synapse_threshold_pvalues", fig)

#%%

x = input_results[input_results["method"] == "Density"].iloc[12]["p_edges_removed"]

fig, ax = plot_thresholding_pvalues(
    input_results, "input_proportion", no_reject_x=x, reject_x=0.28
)


ax.axvline(
    x,
    ax.get_ylim()[0],
    0.95,
    color="black",
    linestyle="--",
    zorder=0,
)
ax.text(
    x + 0.005,
    np.mean(np.sqrt(np.product(ax.get_ylim()))),
    r"$\rightarrow$"
    + "\n\n\n  All tests fail to\n  reject symmetry\n\n\n"
    + r"$\rightarrow$",
    ha="left",
    va="center",
)


leg = ax.get_legend()

gluefig("input_threshold_pvalues_legend", fig)
ax.get_legend().remove()
gluefig("input_threshold_pvalues", fig)

#%%


# fontsize = 10

# int_thresh = SmartSVG(FIG_PATH / "integer_threshold_pvalues.svg")
# int_thresh.set_width(200)
# int_thresh.move(10, 15)
# int_thresh_panel = Panel(
#     int_thresh, Text("A) Synapse count thresholds", 5, 10, size=fontsize, weight="bold")
# )

# input_thresh = SmartSVG(FIG_PATH / "input_threshold_pvalues.svg")
# input_thresh.set_width(200)
# input_thresh.move(10, 15)
# input_thresh_panel = Panel(
#     input_thresh,
#     Text("B) Input proportion thresholds", 5, 10, size=fontsize, weight="bold"),
# )
# input_thresh_panel.move(int_thresh.width * 0.9, 0)

# int_p_removed = SmartSVG(FIG_PATH / "integer_threshold_pvalues_p_removed.svg")
# int_p_removed.set_width(200)
# int_p_removed.move(10, 25)
# int_p_removed_panel = Panel(
#     int_p_removed,
#     Text("C) Synapse count thresholds", 5, 10, size=fontsize, weight="bold"),
#     Text("by edges removed", 15, 20, size=fontsize, weight="bold"),
# )
# int_p_removed_panel.move(0, int_thresh.height * 0.9)

# input_p_removed = SmartSVG(FIG_PATH / "input_threshold_pvalues_p_removed.svg")
# input_p_removed.set_width(200)
# input_p_removed.move(10, 25)
# input_p_removed_panel = Panel(
#     input_p_removed,
#     Text("D) Input proportion thresholds", 5, 10, size=fontsize, weight="bold"),
#     Text("by edges removed", 15, 20, size=fontsize, weight="bold"),
# )
# input_p_removed_panel.move(int_thresh.width * 0.9, int_thresh.height * 0.9)

# fig = Figure(
#     int_thresh.width * 2 * 0.9,
#     int_thresh.height * 2 * 0.95,
#     int_thresh_panel,
#     input_thresh_panel,
#     int_p_removed_panel,
#     input_p_removed_panel,
# )
# fig.save(FIG_PATH / "thresholding_composite.svg")
# fig

#%%
fontsize = 10
methods = SmartSVG(FIG_PATH / "thresholding_methods.svg")
methods.set_width(200)
methods.move(10, 15)
methods_panel = Panel(
    methods, Text("A) Thresholding methods", 5, 10, size=fontsize, weight="bold")
)

synapse_hist = SmartSVG(FIG_PATH / "synapse_weight_histogram.svg")
synapse_hist.set_width(200)
synapse_hist.move(10, 15)
synapse_hist_panel = Panel(
    synapse_hist,
    Text("B) Synapse count distribution", 5, 10, size=fontsize, weight="bold"),
)
synapse_hist_panel.move(0, methods.height * 0.9)

input_hist = SmartSVG(FIG_PATH / "input_proportion_histogram.svg")
input_hist.set_width(200)
input_hist.move(10, 15)
input_hist_panel = Panel(
    input_hist,
    Text("C) Input proportion distribution", 5, 10, size=fontsize, weight="bold"),
)
input_hist_panel.move(synapse_hist.width * 0.85, methods.height * 0.9)

synapse_pvalues = SmartSVG(FIG_PATH / "synapse_threshold_pvalues_legend.svg")
synapse_pvalues.set_width(200)
synapse_pvalues.move(10, 15)
synapse_pvalues_panel = Panel(
    synapse_pvalues,
    Text("D) Synapse thresholding p-values", 5, 10, size=fontsize, weight="bold"),
)
synapse_pvalues_panel.move(0, (methods.height + synapse_hist.height) * 0.9)

input_pvalues = SmartSVG(FIG_PATH / "input_threshold_pvalues_legend.svg")
input_pvalues.set_width(200)
input_pvalues.move(10, 15)
input_pvalues_panel = Panel(
    input_pvalues,
    Text("E) Input thresholding p-values", 5, 10, size=fontsize, weight="bold"),
)
input_pvalues_panel.move(
    synapse_pvalues.width * 0.85, (methods.height + input_hist.height) * 0.9
)


fig = Figure(
    methods.width * 2 * 0.88,
    (methods.height + synapse_hist.height + synapse_pvalues.height) * 0.92,
    methods_panel,
    synapse_hist_panel,
    input_hist_panel,
    synapse_pvalues_panel,
    input_pvalues_panel
    # input_thresh_panel,
    # int_p_removed_panel,
    # input_p_removed_panel,
)
fig.save(FIG_PATH / "thresholding_composite.svg")
fig

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
