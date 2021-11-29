#%%

from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from myst_nb import glue as default_glue
from pkg.data import load_unmatched, load_network_palette, load_node_palette
from pkg.io import savefig
from pkg.plot import set_theme
from seaborn.utils import relative_luminance
from giskard.plot import soft_axis_off

DISPLAY_FIGS = True
FILENAME = "plot_side_layouts"

rng = np.random.default_rng(8888)


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
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
network_palette["L"] = network_palette["Left"]
network_palette["R"] = network_palette["Right"]
node_palette, NODE_KEY = load_node_palette()

left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

#%%

from graspologic.plot import networkplot

from graspologic.embed import AdjacencySpectralEmbed

ase = AdjacencySpectralEmbed(n_components=24, check_lcc=False, concat=True)
left_ase_embedding = ase.fit_transform(left_adj)
right_ase_embedding = ase.fit_transform(right_adj)

from umap import UMAP

umapper = UMAP(
    n_components=2,
    n_neighbors=64,
    min_dist=0.8,
    metric="cosine",
    random_state=rng.integers(np.iinfo(np.int32).max),
)
left_umap_embedding = umapper.fit_transform(left_ase_embedding)
right_umap_embedding = umapper.fit_transform(right_ase_embedding)

#%%

networkplot_kws = dict(
    x="x",
    y="y",
    edge_linewidth=0.15,
    edge_alpha=0.2,
    node_hue="hemisphere",
    palette=network_palette,
    edge_hue="source",
    node_size="degree",
    node_sizes=(15, 100),
)


def soft_axis_off(ax, top=False, bottom=False, left=False, right=False):
    # ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)


fig, axs = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw=dict(wspace=0))

ax = axs[0]
soft_axis_off(ax)

left_nodes["degree"] = left_adj.sum(axis=0) + left_adj.sum(axis=1)
left_nodes["x"] = left_umap_embedding[:, 0]
left_nodes["y"] = left_umap_embedding[:, 1]

networkplot(left_adj, node_data=left_nodes.reset_index(), ax=ax, **networkplot_kws)

ax.set_xlabel("Left", color=network_palette["Left"], fontsize=60)
ax.set_ylabel("")

ax = axs[1]
soft_axis_off(ax)

right_nodes["x"] = right_umap_embedding[:, 0]
right_nodes["y"] = right_umap_embedding[:, 1]
right_nodes["degree"] = right_adj.sum(axis=0) + right_adj.sum(axis=1)

networkplot(right_adj, node_data=right_nodes.reset_index(), ax=ax, **networkplot_kws)

ax.set_xlabel("Right", color=network_palette["Right"], fontsize=60)
ax.set_ylabel("")

fig.set_facecolor("white")

gluefig("2_network_layout", fig)
