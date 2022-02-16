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
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.perturb import remove_edges
from pkg.plot import set_theme
from pkg.stats import stochastic_block_test


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
from graspologic.plot import networkplot
from pkg.plot import networkplot_grouped, heatmap_grouped
import matplotlib as mpl
from giskard.plot import merge_axes


np.random.seed(888888)
ns = [5, 6, 7]
B = np.array([[0.8, 0.2, 0.05], [0.05, 0.9, 0.2], [0.05, 0.05, 0.7]])
A1, labels = sbm(ns, B, directed=True, loops=False, return_labels=True)

node_data = pd.DataFrame(index=np.arange(A1.shape[0]))
node_data["labels"] = labels + 1
palette = dict(zip(np.unique(labels) + 1, sns.color_palette("Set2")[3:]))


fig, axs = plt.subplots(
    1, 4, figsize=(13, 4), gridspec_kw=dict(width_ratios=[1, 0.5, 0.5, 1])
)
ax = axs[0]
node_data = networkplot_grouped(A1, node_data, palette=palette, ax=ax)

n_select = 10
row_inds, col_inds = np.nonzero(A1)
np.random.seed(8888)
choice_inds = np.random.choice(len(row_inds), size=n_select)
for i in choice_inds:
    source_node = row_inds[i]
    target_node = col_inds[i]
    x1, y1 = node_data.loc[source_node, ["x", "y"]]
    x2, y2 = node_data.loc[target_node, ["x", "y"]]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    ax.text(
        x,
        y,
        "x",
        va="center",
        ha="center",
        color="darkred",
        fontsize="medium",
        zorder=2,
    )


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
top_ax = heatmap_grouped(Bhat1, [1, 2, 3], palette=palette, ax=ax)
top_ax.set_title("Adjust connection probabilities", fontsize="medium", x=1.2, y=6)


ax = axs[2]
Bhat1 = misc["probabilities1"].values
top_ax = heatmap_grouped(0.6 * Bhat1, [1, 2, 3], palette=palette, ax=ax)

ax.annotate(
    "",
    xy=(0, 1.5),
    xytext=(-1, 1.5),
    arrowprops=dict(
        arrowstyle="simple",
        shrinkB=9,
        facecolor="black",
    ),
    zorder=1,
)

ax = axs[3]
ax.set_title("Rerun SBM testing")
ax.axis("off")

fig.set_facecolor("w")

gluefig("adjusted_methods_explain", fig)


# vmin = 0
# vmax = 1

# cmap = mpl.cm.get_cmap("Blues")
# normlize = mpl.colors.Normalize(vmin, vmax)
# cmin, cmax = normlize([vmin, vmax])
# cc = np.linspace(cmin, cmax, 256)
# cmap = mpl.colors.ListedColormap(cmap(cc))

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
stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    null_odds=1,
    combine_method="tippett",
)
pvalue_vmin = np.log10(np.nanmin(misc["uncorrected_pvalues"].values))

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


#%%

set_theme(font_scale=1.25)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(
    data=resample_results[resample_results["combine_method"] == "tippett"],
    x="pvalue",
    ax=ax,
    color=neutral_color,
    # bins=30,
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

median_resample_pvalue = np.median(resample_results["pvalue"])

colors = sns.color_palette("Set2")

from matplotlib.patheffects import Stroke, Normal


def nice_text(
    x,
    y,
    s,
    ax=None,
    color="black",
    fontsize=None,
    transform=None,
    ha="left",
    va="center",
    linewidth=4,
    linecolor="black",
):
    if transform is None:
        transform = ax.transData
    text = ax.text(
        x,
        y,
        s,
        color=color,
        fontsize=fontsize,
        transform=transform,
        ha=ha,
        va=va,
    )
    text.set_path_effects([Stroke(linewidth=linewidth, foreground=linecolor), Normal()])


color = colors[3]
ax.axvline(median_resample_pvalue, color=color, linewidth=3)
# ax.text(
#     median_resample_pvalue - 0.0025,
#     ylim[1] * 0.9,
#     f"Median = {median_resample_pvalue:0.2g}",
#     ha="right",
#     color=color,
# )
ax.text(
    median_resample_pvalue - 0.0025,
    ylim[1] * 0.9,
    f"Median = {median_resample_pvalue:0.2g}",
    color=color,
    ha="right",
)

color = colors[4]
ax.axvline(pvalue, 0, 0.5, color=color, linewidth=3)
ax.text(
    pvalue - 0.0002,
    ylim[1] * 0.43,
    f"Analytic = {pvalue:0.2g}",
    ha="right",
    color=color,
)

gluefig("resampled_pvalues_distribution", fig)

#%%

from pkg.plot import plot_pvalues

# TODO get the actual pvalue vmin

fig, axs = plot_pvalues(misc, pvalue_vmin)

gluefig("sbm_uncorrected_pvalues", fig)

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
total_height = 700

FIG_PATH = Path("bilateral-connectome/results/figs")
FIG_PATH = FIG_PATH / FILENAME

fontsize = 35

sbm_methods_explain_svg = SVG(FIG_PATH / "adjusted_methods_explain.svg")
sbm_methods_explain_svg_scaler = 1 / sbm_methods_explain_svg.height * total_height / 2
sbm_methods_explain_svg = sbm_methods_explain_svg.scale(sbm_methods_explain_svg_scaler)

sbm_methods_explain = Panel(
    sbm_methods_explain_svg,
    Text("A)", 5, 20, size=fontsize, weight="bold"),
)


resampled_pvalues_distribution = SVG(FIG_PATH / "resampled_pvalues_distribution.svg")
resampled_pvalues_distribution.scale(
    1 / resampled_pvalues_distribution.height * total_height / 2
)

resampled_pvalues_distribution = Panel(
    resampled_pvalues_distribution,
    Text("B)", 5, 20, size=fontsize, weight="bold"),
).move(0, 300)


pvalues = SVG(FIG_PATH / "sbm_uncorrected_pvalues.svg")
pvalues.scale(1 / pvalues.height * total_height / 2)

pvalues = Panel(
    pvalues.move(50, 30),
    Text("C)", 5, 20, size=fontsize, weight="bold"),
).move(400, 300)

fig = Figure(850, 625, sbm_methods_explain, resampled_pvalues_distribution, pvalues)
fig.save(FIG_PATH / "adjusted_sbm_composite.svg")
fig


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
