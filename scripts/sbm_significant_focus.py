#%% [markdown]
# # Focus on the significant group-to-group connections

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

left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))

#%%

stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    combine_method="fisher",
)
glue("uncorrected_pvalue", pvalue)
n_tests = misc["n_tests"]
glue("n_tests", n_tests)
print(pvalue)

#%%
misc["uncorrected_pvalues"]

from statsmodels.stats.multitest import multipletests

pvalues = misc["uncorrected_pvalues"].values.copy()
inds = np.nonzero(~np.isnan(pvalues))

reject, pvals_corrected, _, _ = multipletests(
    pvalues[inds], alpha=0.05, method="bonferroni"
)

#%%
source_reject_inds = inds[0][reject]
target_reject_inds = inds[1][reject]
index = misc["uncorrected_pvalues"].index
list(zip(index[source_reject_inds], index[target_reject_inds]))

#%%

from pkg.stats import rdpg_test
from pkg.utils import get_seeds


seeds = get_seeds(left_nodes, right_nodes)

n_components = 8
stat, pvalue, misc = rdpg_test(
    left_adj, right_adj, seeds=seeds, n_components=n_components
)

#%%
Z_left = misc["Z1"]
Z_right = misc["Z2"]

#%%

# def plot_latents(
#     left,
#     right,
#     title="",
#     n_show=4,
#     alpha=0.6,
#     linewidth=0.4,
#     s=10,
#     connections=False,
#     palette=None,
# ):
#     if n_show > left.shape[1]:
#         n_show = left.shape[1]
#     plot_data = np.concatenate([left, right], axis=0)
#     labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
#     pg = pairplot(
#         plot_data[:, :n_show],
#         labels=labels,
#         title=title,
#         size=s,
#         palette=palette,
#     )

#     # pg._legend.remove()
#     return pg

# plot_latents(Z1, Z2)


from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull


def fit_bounding_contour(points, s=0, per=1):
    hull = ConvexHull(points)
    boundary_indices = list(hull.vertices)
    boundary_indices.append(boundary_indices[0])
    boundary_points = points[boundary_indices].copy()

    tck, u = splprep(boundary_points.T, u=None, s=s, per=per)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new


def draw_bounding_contour(points, color=None, linewidth=2):
    x_new, y_new = fit_bounding_contour(points)
    ax.plot(x_new, y_new, color=color, zorder=-1, linewidth=linewidth, alpha=0.5)
    ax.fill(x_new, y_new, color=color, zorder=-2, alpha=0.1)


Z = Z_left
nodes = left_nodes
side = "Left"
label = "KCs"


def draw_selected_nodes(
    points,
    nodes,
    label,
    dims=(0, 1),
    select_size=20,
    select_alpha=1,
    color=None,
    ax=None,
):
    mask = nodes[GROUP_KEY] == label
    select_points = points[mask]

    network_palette[side]
    nonselect_size = 2
    nonselect_alpha = 0.1

    sns.scatterplot(
        x=points[mask, dims[0]],
        y=points[mask, dims[1]],
        # hue=nodes[GROUP_KEY],
        # palette=node_palette,
        color=color,
        ax=ax,
        s=select_size,
        alpha=select_alpha,
    )
    draw_bounding_contour(select_points[:, dims], color=color)

    ax.set(xticks=[], yticks=[])
    ax.spines[["top", "right"]].set_visible(True)

    sns.scatterplot(
        x=points[~mask, dims[0]],
        y=points[~mask, dims[1]],
        # hue=nodes[GROUP_KEY],
        # palette=node_palette,
        color=color,
        ax=ax,
        s=nonselect_size,
        alpha=nonselect_alpha,
    )


def set_bounds(points1, points2, dims, ax):
    points = np.concatenate((points1, points2), axis=0)
    xmin = points[:, dims[0]].min()
    xmax = points[:, dims[0]].max()
    ymin = points[:, dims[1]].min()
    ymax = points[:, dims[1]].max()

    xpad = 0.25 * (xmax - xmin)
    xlim = np.array((xmin - xpad, xmax + xpad))
    ypad = 0.25 * (ymax - ymin)
    ylim = np.array((ymin - ypad, ymax + ypad))
    ax.set(xlim=xlim, ylim=ylim)


max_dim = 6
fig, axs = plt.subplots(max_dim - 1, max_dim - 1, figsize=(20, 20))

for i in range(0, max_dim):
    for j in range(i + 1, max_dim):
        ax = axs[i, j - 1]
        dims = (i, j)
        draw_selected_nodes(
            Z_left, left_nodes, "KCs", dims=dims, color=network_palette["Left"], ax=ax
        )
        draw_selected_nodes(
            Z_right,
            right_nodes,
            "KCs",
            dims=dims,
            color=network_palette["Right"],
            ax=ax,
        )


# remove dummy axes
for ax in axs.flat:
    if not ax.has_data():
        ax.set_visible(False)

plt.tight_layout()
