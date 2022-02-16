#%% [markdown]
# # SBM test with density adjustment

#%%
from graspologic.plot import networkplot
from pkg.io.io import FIG_PATH, OUT_PATH
from pkg.plot.bound import bound_points
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


DISPLAY_FIGS = True

FILENAME = "adjusted_sbm_unmatched_test"


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

#%%

from graspologic.simulations import sbm
from graspologic.plot import networkplot, heatmap, adjplot
import networkx as nx
from pkg.plot import bound_points
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.colors import ListedColormap

np.random.seed(888888)
ns = [5, 6, 7]
B = np.array([[0.8, 0.2, 0.05], [0.05, 0.9, 0.2], [0.05, 0.05, 0.7]])
A1, labels = sbm(ns, B, directed=True, loops=False, return_labels=True)

node_data = pd.DataFrame(index=np.arange(A1.shape[0]))
node_data["labels"] = labels + 1
palette = dict(zip(np.unique(labels) + 1, sns.color_palette("Set2")[3:]))




def remove_shared_ax(ax):
    """
    Remove ax from its sharex and sharey
    """
    # Remove ax from the Grouper object
    shax = ax.get_shared_x_axes()
    shay = ax.get_shared_y_axes()
    shax.remove(ax)
    shay.remove(ax)

    # Set a new ticker with the respective new locator and formatter
    for axis in [ax.xaxis, ax.yaxis]:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        # No ticks and no labels
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)




def heatmap_grouped(Bhat, palette=None, ax=None, pad=0, color_size="5%"):

    heatmap(Bhat, ax=ax, cmap="Blues", vmin=0, vmax=1, center=None, cbar=False)
    divider = make_axes_locatable(ax)
    top_ax = divider.append_axes("top", size=color_size, pad=pad, sharex=ax)
    remove_shared_ax(top_ax)
    draw_colors(top_ax, "x", labels=[1, 2, 3], palette=palette)
    color_ax = divider.append_axes("left", size=color_size, pad=pad, sharex=ax)
    remove_shared_ax(color_ax)
    draw_colors(color_ax, "y", labels=[1, 2, 3], palette=palette)
    return top_ax


fig, axs = plt.subplots(1, 4, figsize=(16, 4))
ax = axs[0]
networkplot_grouped(A1, node_data, palette=palette, ax=ax)
ax.set_title("Randomly subsample\nedges", fontsize="medium")
ax.set_ylabel(
    "Right",
    color=network_palette["Right"],
    size="large",
    rotation=0,
    ha="right",
    labelpad=10,
)


ax = axs[1]
_, _, misc = stochastic_block_test(A1, A1, node_data["labels"], node_data["labels"])
Bhat1 = misc["probabilities1"].values
top_ax = heatmap_grouped(Bhat1, palette=palette, ax=ax)
top_ax.set_title("Fit stochastic\nblock models", fontsize="medium")

fig.set_facecolor("w")

from giskard.plot import merge_axes

vmin = 0
vmax = 1

cmap = mpl.cm.get_cmap("Blues")
normlize = mpl.colors.Normalize(vmin, vmax)
cmin, cmax = normlize([vmin, vmax])
cc = np.linspace(cmin, cmax, 256)
cmap = mpl.colors.ListedColormap(cmap(cc))

# ax = merge_axes(fig, axs, rows=None, cols=2)
# node_sizes=(20, 200),
#         node_kws=dict(linewidth=1, edgecolor="black"),
#         node_alpha=1.0,
#         edge_kws=dict(color="black"),


# def compare_probability_row(source, target, y):

#     x1 = 0.1
#     x2 = 0.25
#     sns.scatterplot(
#         x=[x1, x2],
#         y=[y, y],
#         hue=[source, target],
#         linewidth=1,
#         edgecolor="black",
#         palette=palette,
#         ax=ax,
#         legend=False,
#         s=100,
#     )
#     # ax.arrow(x1, y, x2 - x1, 0, arrowprops=dict(arrowstyle='->'))
#     ax.annotate(
#         "",
#         xy=(x2, y),
#         xytext=(x1, y),
#         arrowprops=dict(
#             arrowstyle="simple",
#             connectionstyle="arc3,rad=-0.7",
#             facecolor="black",
#             shrinkA=5,
#             shrinkB=5,
#             # mutation_scale=,
#         ),
#     )

#     x3 = 0.4

#     size = 0.1
#     phat = Bhat1[source - 1, target - 1]
#     color = cmap(phat)
#     patch = mpl.patches.Rectangle(
#         (x3, y - size / 4), width=size, height=size / 2, facecolor=color
#     )
#     ax.add_patch(patch)

#     text = ax.text(0.645, y, "?", ha="center", va="center")
#     text.set_bbox(dict(facecolor="white", edgecolor="white"))

#     x4 = 0.8
#     phat = Bhat2[source - 1, target - 1]
#     color = cmap(phat)
#     patch = mpl.patches.Rectangle(
#         (x4, y - size / 4), width=size, height=size / 2, facecolor=color
#     )
#     ax.add_patch(patch)

#     ax.plot([x3, x4], [y, y], linewidth=2.5, linestyle=":", color="grey", zorder=-1)


# ax.text(0.4, 0.93, r"$\hat{B}^{(L)}$", color=network_palette["Left"])
# ax.text(0.8, 0.93, r"$\hat{B}^{(R)}$", color=network_palette["Right"])
# compare_probability_row(1, 1, 0.9)
# compare_probability_row(1, 2, 0.85)
# compare_probability_row(1, 3, 0.8)
# compare_probability_row(2, 1, 0.75)
# compare_probability_row(2, 2, 0.7)
# compare_probability_row(2, 3, 0.65)
# compare_probability_row(3, 1, 0.6)
# compare_probability_row(3, 2, 0.55)
# compare_probability_row(3, 3, 0.5)


# ax.annotate(
#     "",
#     xy=(0.645, 0.48),
#     xytext=(0.5, 0.41),
#     arrowprops=dict(
#         arrowstyle="simple",
#         facecolor="black",
#     ),
# )
# y = 0.32
# ax.text(0.2, y, r"$H_0$:", fontsize="large")
# ax.text(0.42, y, r"$\hat{B}^{L}_{ij}$", color=network_palette["Left"], fontsize="large")
# ax.text(0.55, y, r"$=$", fontsize="large")
# ax.text(0.7, y, r"$\hat{B}^{R}_{ij}$", color=network_palette["Right"], fontsize="large")
# y = y - 0.1
# ax.text(0.2, y, r"$H_A$:", fontsize="large")
# ax.text(0.42, y, r"$\hat{B}^{L}_{ij}$", color=network_palette["Left"], fontsize="large")
# ax.text(0.55, y, r"$\neq$", fontsize="large")
# ax.text(0.7, y, r"$\hat{B}^{R}_{ij}$", color=network_palette["Right"], fontsize="large")
# patch = mpl.patches.Rectangle(
#     xy=(0.18, y - 0.03),
#     width=0.7,
#     height=0.21,
#     facecolor="white",
#     edgecolor="lightgrey",
# )
# ax.add_patch(patch)


# ax.set_title("Compare estimated\nprobabilities", fontsize="medium")
# ax.set(xlim=(0, 1), ylim=(0.18, 1))
# ax.axis("off")

# ax = merge_axes(fig, axs, rows=None, cols=3)
# ax.axis("off")
# ax.set_title("Combine p-values\nfor overall test", fontsize="medium")

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
# y = 0.32
# ax.text(0.2, y, r"$H_0$:", fontsize="large")
# ax.text(0.42, y, r"$\hat{B}^{L}$", color=network_palette["Left"], fontsize="large")
# ax.text(0.55, y, r"$=$", fontsize="large")
# ax.text(0.7, y, r"$\hat{B}^{R}$", color=network_palette["Right"], fontsize="large")
# y = y - 0.1
# ax.text(0.2, y, r"$H_A$:", fontsize="large")
# ax.text(0.42, y, r"$\hat{B}^{L}$", color=network_palette["Left"], fontsize="large")
# ax.text(0.55, y, r"$\neq$", fontsize="large")
# ax.text(0.7, y, r"$\hat{B}^{R}$", color=network_palette["Right"], fontsize="large")
# patch = mpl.patches.Rectangle(
#     xy=(0.18, y - 0.03),
#     width=0.7,
#     height=0.21,
#     facecolor="white",
#     edgecolor="lightgrey",
# )
# ax.add_patch(patch)


gluefig("adjusted_methods_explain", fig)


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
from tqdm import tqdm
from pathlib import Path

rows = []
n_resamples = 500
glue("n_resamples", n_resamples)
RERUN_SIM = False

OUT_PATH = Path(f"bilateral-connectome/results/outputs/{FILENAME}")

if RERUN_SIM:
    for i in tqdm(range(n_resamples)):
        subsampled_right_adj = remove_edges(
            right_adj, effect_size=n_remove, random_seed=rng
        )
        for combine_method in ["tippett"]:
            stat, pvalue, misc = stochastic_block_test(
                left_adj,
                subsampled_right_adj,
                labels1=left_labels,
                labels2=right_labels,
                method="fisher",
                combine_method=combine_method,
            )
            rows.append(
                {
                    "stat": stat,
                    "pvalue": pvalue,
                    "misc": misc,
                    "resample": i,
                    "combine_method": combine_method,
                }
            )
    resample_results = pd.DataFrame(rows)
    resample_results.to_csv(OUT_PATH / "resample_results.csv")
else:
    resample_results = pd.read_csv(OUT_PATH / "resample_results.csv", index_col=0)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(
    data=resample_results[resample_results["combine_method"] == "tippett"],
    x="pvalue",
    ax=ax,
    color=neutral_color,
    bins=30,
    kde=True,
    # kde_kws=dict(clip=[0, 1]),
    log_scale=True,
    stat="density",
)
ax.set(xlabel="p-value", ylabel="", yticks=[])
ax.spines["left"].set_visible(False)
ax.axvline(0.05, linestyle=":", color="black")
ylim = ax.get_ylim()
ax.text(0.06, ylim[1] * 0.9, r"$\alpha = 0.05$")

mean_resample_pvalue = np.mean(resample_results["pvalue"])
median_resample_pvalue = np.median(resample_results["pvalue"])
ax.axvline(median_resample_pvalue, color="darkred", linewidth=2)
ax.text(
    median_resample_pvalue - 0.0025,
    ylim[1] * 0.9,
    f"Median = {median_resample_pvalue:0.2f}",
    ha="right",
    color="darkred",
)
gluefig("resampled_pvalues_distribution", fig)


#%%
null_odds = density_left / density_right
stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    null_odds=null_odds,
    combine_method="tippett",
)
glue("corrected_pvalue", pvalue)
print(pvalue)
print(f"{pvalue:.2g}")

# #%%
# left_nodes["inds"] = range(len(left_nodes))
# sub_left_nodes = left_nodes[left_nodes[GROUP_KEY] != "KCs"]
# sub_left_inds = sub_left_nodes["inds"].values
# right_nodes["inds"] = range(len(right_nodes))
# sub_right_nodes = right_nodes[right_nodes[GROUP_KEY] != "KCs"]
# sub_right_inds = sub_right_nodes["inds"].values

# sub_left_adj = left_adj[np.ix_(sub_left_inds, sub_left_inds)]
# sub_right_adj = right_adj[np.ix_(sub_right_inds, sub_right_inds)]
# sub_left_labels = sub_left_nodes[GROUP_KEY]
# sub_right_labels = sub_right_nodes[GROUP_KEY]

# from pkg.stats import erdos_renyi_test

# stat, pvalue, misc = erdos_renyi_test(sub_left_adj, sub_right_adj)
# print(pvalue)

# stat, pvalue, misc = stochastic_block_test(
#     sub_left_adj,
#     sub_right_adj,
#     labels1=sub_left_labels,
#     labels2=sub_right_labels,
#     method="fisher",
#     combine_method="tippett",
# )
# print(pvalue)

# n_edges_left = np.count_nonzero(sub_left_adj)
# n_edges_right = np.count_nonzero(sub_right_adj)
# n_left = sub_left_adj.shape[0]
# n_right = sub_right_adj.shape[0]
# density_left = n_edges_left / (n_left ** 2)
# density_right = n_edges_right / (n_right ** 2)

# null_odds = density_left / density_right
# stat, pvalue, misc = stochastic_block_test(
#     sub_left_adj,
#     sub_right_adj,
#     labels1=sub_left_labels,
#     labels2=sub_right_labels,
#     method="fisher",
#     null_odds=null_odds,
#     combine_method="tippett",
# )
# print(pvalue)


#%%
from svgutils.compose import Figure, Panel, SVG, Text
from pathlib import Path


total_width = 1000
total_height = 1500

FIG_PATH = Path("bilateral-connectome/results/figs")
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
