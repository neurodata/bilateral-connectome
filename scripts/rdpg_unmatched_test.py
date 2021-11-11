#%% [markdown]
# # An embedding-based test

#%%

from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.embed import select_svd
from graspologic.plot import pairplot
from graspologic.utils import augment_diagonal
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import rdpg_test
from pkg.utils import get_seeds

DISPLAY_FIGS = False
FILENAME = "rdpg_unmatched_test"

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
node_palette, NODE_KEY = load_node_palette()

left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))

seeds = get_seeds(left_nodes, right_nodes)

#%% [markdown]
# ## A test based on latent positions

#%% [markdown]
# ### Look at the singular values
#%%


def screeplot(sing_vals, elbow_inds=None, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    if elbow_inds is not None:
        plt.scatter(
            elbow_inds,
            sing_vals[elbow_inds - 1],
            marker="x",
            s=50,
            zorder=10,
            color=color,
        )
    ax.set(ylabel="Singular value", xlabel="Index")
    return ax


max_n_components = 64
_, left_singular_values, _ = select_svd(
    augment_diagonal(left_adj), n_elbows=6, n_components=max_n_components
)
_, right_singular_values, _ = select_svd(
    augment_diagonal(right_adj), n_elbows=6, n_components=max_n_components
)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
screeplot(
    left_singular_values,
    color=network_palette["Left"],
    ax=ax,
    label="Left",
)
screeplot(
    right_singular_values,
    color=network_palette["Right"],
    ax=ax,
    label="Right",
)
ax.legend()

gluefig("screeplot", fig)

#%% [markdown]
#
# ```{glue:figure} fig:rdpg_unmatched_test-screeplot
# :name: "fig:rdpg_unmatched_test-screeplot"

# Comparison of the singular values from the spectral decompositions of the left and
# right hemisphere adjacency matrices. Note that the right hemisphere singular values
# tend to be slightly higher than the corresponding singular value on the left
# hemisphere, which is consistent with an increased density on the right hemisphere as
# seen in [](er_unmatched_test.ipynb).
# ```


#%% [markdown]
# ### Run the test
#%%
n_components = 8  # TODO trouble is that this choice is somewhat arbitrary...
stat, pvalue, misc = rdpg_test(
    left_adj, right_adj, seeds=seeds, n_components=n_components
)
glue("pvalue", pvalue)

#%% [markdown]
# ### Look at the embeddings
#%%


Z1 = misc["Z1"]
Z2 = misc["Z2"]


def plot_latents(
    left,
    right,
    title="",
    n_show=4,
    alpha=0.6,
    linewidth=0.4,
    s=10,
    connections=False,
    palette=None,
):
    if n_show > left.shape[1]:
        n_show = left.shape[1]
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(
        plot_data[:, :n_show],
        labels=labels,
        title=title,
        size=s,
        palette=palette,
    )

    # pg._legend.remove()
    return pg


n_show = 4
pg = plot_latents(Z1, Z2, palette=network_palette, n_show=n_show)
fig = pg.figure
eff_n_components = Z1.shape[1]
glue("n_show", n_show)
glue("eff_n_components", eff_n_components)
gluefig("latents", fig)


#%% [markdown]

# ```{glue:figure} fig:rdpg_unmatched_test-latents
# :name: "fig:rdpg_unmatched_test-latents"

# Comparison of the latent positions used for the test based on the random dot product
# graph. This plot shows only the first {glue:text}`rdpg_unmatched_test-n_show`
# dimensions, though the test was run in {glue:text}`rdpg_unmatched_test-eff_n_components`.
# The p-value for the test comparing the multivariate distribution of latent positions
# for the left vs. the right hemispheres (distance correlation 2-sample test) is
# {glue:text}`rdpg_unmatched_test-pvalue:.2f`, indicating that we fail to reject our
# null hypothesis of bilateral symmetry under this null model.
# ```
