#%% [markdown]
# # An embedding-based test

#%%

from hyppo.tools.indep_sim import square
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

DISPLAY_FIGS = True
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
    plt.plot(
        range(1, len(sing_vals) + 1), sing_vals, color=color, label=label, linewidth=3
    )
    if elbow_inds is not None:
        elbow_inds = np.array(elbow_inds)
        plt.scatter(
            elbow_inds,
            sing_vals[elbow_inds - 1],
            marker=".",
            s=300,
            zorder=10,
            color=color,
        )
        ylim = ax.get_ylim()
        for ind in elbow_inds:
            ax.plot([ind, ind], [0, sing_vals[ind - 1]], color="grey", linestyle=":")
        ax.set_ylim(ylim)
    ax.set(ylabel="Singular value", xlabel="Index")
    return ax


max_n_components = 25
_, left_singular_values, _ = select_svd(
    augment_diagonal(left_adj), n_elbows=6, n_components=max_n_components
)
_, right_singular_values, _ = select_svd(
    augment_diagonal(right_adj), n_elbows=6, n_components=max_n_components
)

from graspologic.embed import select_dimension

left_elbow_inds, left_elbow_pos = select_dimension(
    augment_diagonal(left_adj), n_elbows=4
)
right_elbow_inds, right_elbow_pos = select_dimension(
    augment_diagonal(right_adj), n_elbows=4
)

xticks = list(np.union1d(left_elbow_inds, right_elbow_inds))
xticks += [15, 20, 25]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
screeplot(
    right_singular_values,
    color=network_palette["Right"],
    elbow_inds=right_elbow_inds,
    ax=ax,
    label="Right",
)
screeplot(
    left_singular_values,
    color=network_palette["Left"],
    elbow_inds=left_elbow_inds,
    ax=ax,
    label="Left",
)
ax.set(xticks=xticks)
ax.legend()
ax.yaxis.set_major_locator(plt.MaxNLocator(3))

gluefig("screeplot", fig)

print(left_elbow_inds)
print(right_elbow_inds)

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
print(pvalue)

#%%
_, norm_pvalue, _ = rdpg_test(
    left_adj, right_adj, seeds=seeds, n_components=n_components, normalize_nodes=True
)
glue("norm_pvalue", norm_pvalue)
print(norm_pvalue)

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

n_show = 3
pg = plot_latents(Z1, Z2, palette=network_palette, n_show=n_show, s=20)
fig = pg.figure
gluefig(f"latents_d={n_show}", fig)

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

#%%

max_n_components = 16

from tqdm import tqdm

n_tests = max_n_components * (max_n_components - 1) / 2

rows = []
with tqdm(total=n_tests) as pbar:
    for align_n_components in range(1, max_n_components + 1):
        for test_n_components in range(1, align_n_components + 1):
            stat, pvalue, _ = rdpg_test(
                left_adj,
                right_adj,
                seeds=seeds,
                n_components=test_n_components,
                align_n_components=align_n_components,
            )
            rows.append(
                {
                    "stat": stat,
                    "pvalue": pvalue,
                    "align_n_components": align_n_components,
                    "test_n_components": test_n_components,
                }
            )
            pbar.update(1)

results = pd.DataFrame(rows)

#%%
square_results = results.pivot(
    index="test_n_components", columns="align_n_components", values="pvalue"
)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_results = np.log10(square_results)

annot = np.full((max_n_components, max_n_components), "")
annot[square_results < 0.05] = "X"
sns.heatmap(
    plot_results,
    ax=ax,
    square=True,
    cmap="RdBu",
    center=0,
    cbar_kws=dict(shrink=0.6),
    annot=annot,
    fmt="s",
)
ax.set_xlabel("# dimensions for alignment", fontsize="large")
ax.set_ylabel("# dimensions for testing", fontsize="large")
ax.set_xticks(np.array([1, 4, 8, 12, 16]) - 0.5)
ax.set_xticklabels(np.array([1, 4, 8, 12, 16]), rotation=0)
ax.set_yticks(np.array([1, 4, 8, 12, 16]) - 0.5)
ax.set_yticklabels(np.array([1, 4, 8, 12, 16]), rotation=0)
ax.set_title(r"$log_{10}($p-value$)$", fontsize="x-large")

cax = fig.axes[1]
cax.get_ylim()
cax.plot(
    [0, 1], [np.log10(0.05), np.log10(0.05)], zorder=100, color="black", linewidth=3
)

import matplotlib.transforms as mtransforms

trans = mtransforms.blended_transform_factory(cax.transData, cax.transAxes)
cax.annotate(
    r"$\alpha$ = 0.05",
    (0.93, np.log10(0.05)),
    xytext=(30, -10),
    textcoords="offset points",
    va="center",
    arrowprops={"arrowstyle": "-", "linewidth": 2, "relpos": (0, 0.5)},
)

pos = (8, 8)


def annotate_pos(pos, xytext):
    val = square_results.loc[pos]
    ax.annotate(
        f"{val:0.2g}",
        (pos[0] - 0.8, pos[0] - 0.2),
        ha="right",
        textcoords="offset points",
        xytext=xytext,
        arrowprops={"arrowstyle": "-", "linewidth": 2, "relpos": (1, 0.5)},
    )


annotate_pos((8, 8), (-25, -20))
annotate_pos((9, 9), (-25, -35))


gluefig("pvalue_dimension_matrix", fig)
