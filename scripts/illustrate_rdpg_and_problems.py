#%% [markdown]
# # Look at a latent distribution test and some issues with it

#%% [markdown]
# ## Preliminaries
#%%

import datetime
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
from giskard.align import joint_procrustes
from graspologic.embed import (
    AdjacencySpectralEmbed,
    select_dimension,
)
from graspologic.utils import augment_diagonal, binarize, pass_to_ranks
from hyppo.ksample import KSample
from pkg.data import (
    load_maggot_graph,
    load_network_palette,
    load_node_palette,
    select_nice_nodes,
)
from pkg.io import savefig
from pkg.plot import set_theme
import pandas as pd
import seaborn as sns

from matplotlib.ticker import MaxNLocator
from graspologic.plot import pairplot


def stashfig(name, **kwargs):
    foldername = "illustrate_rdpg_and_problems"
    savefig(name, foldername=foldername, **kwargs)


# %% [markdown]
# ## Load and process data
#%%

t0 = time.time()
set_theme()

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True)
ll_adj = left_mg.sum.adj
rr_adj = right_mg.sum.adj

#%%
valid_pairs = np.intersect1d(left_mg.nodes["pair_id"], right_mg.nodes["pair_id"])
valid_pairs = valid_pairs[1:]

left_nodes = left_mg.nodes
left_mg.nodes["_inds"] = np.arange(len(left_nodes))
right_nodes = right_mg.nodes
right_mg.nodes["_inds"] = np.arange(len(right_nodes))

left_paired_nodes = left_nodes[left_nodes["pair_id"].isin(valid_pairs)]
right_paired_nodes = right_nodes[right_nodes["pair_id"].isin(valid_pairs)]
left_paired_inds = left_paired_nodes.sort_values("pair_id")["_inds"]
right_paired_inds = right_paired_nodes.sort_values("pair_id")["_inds"]


#%% [markdown]
# ## Embed the network using adjacency spectral embedding

#%%
def preprocess_for_embed(ll_adj, rr_adj, preprocess):
    if "binarize" in preprocess:
        ll_adj_to_embed = binarize(ll_adj)
        rr_adj_to_embed = binarize(rr_adj)

    if "rescale" in preprocess:
        ll_norm = np.linalg.norm(ll_adj_to_embed, ord="fro")
        rr_norm = np.linalg.norm(rr_adj_to_embed, ord="fro")
        mean_norm = (ll_norm + rr_norm) / 2
        ll_adj_to_embed *= mean_norm / ll_norm
        rr_adj_to_embed *= mean_norm / rr_norm
    return ll_adj_to_embed, rr_adj_to_embed


def embed(adj, n_components=40, ptr=False):
    if ptr:
        adj = pass_to_ranks(adj)
    elbow_inds, _ = select_dimension(augment_diagonal(adj), n_elbows=5)
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(adj)
    return out_latent, in_latent, ase.singular_values_, elbow_inds


preprocess = ["binarize", "rescale"]
ll_adj, rr_adj = preprocess_for_embed(ll_adj, rr_adj, preprocess)

#%%


def embed_and_align(
    ll_adj, rr_adj, align_n_components=None, n_components=8, align=True
):
    if align_n_components is None:
        align_n_components = n_components

    X_ll, Y_ll, _, _ = embed(ll_adj, n_components=align_n_components)
    X_rr, Y_rr, _, _ = embed(rr_adj, n_components=align_n_components)
    if align:
        X_ll, Y_ll = joint_procrustes(
            (X_ll, Y_ll),
            (X_rr, Y_rr),
            method="seeded",
            seeds=(left_paired_inds, right_paired_inds),
        )
    left_composite_latent = np.concatenate(
        (X_ll[:, :n_components], Y_ll[:, :n_components]), axis=1
    )
    right_composite_latent = np.concatenate(
        (X_rr[:, :n_components], Y_rr[:, :n_components]), axis=1
    )

    return left_composite_latent, right_composite_latent


n_components = 21
X_ll, Y_ll, left_sing_vals, left_elbow_inds = embed(ll_adj, n_components=n_components)
X_rr, Y_rr, right_sing_vals, right_elbow_inds = embed(rr_adj, n_components=n_components)

#%% [markdown]
# ### Plot a screeplot for the embeddings
#%%


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    plt.scatter(
        range(1, len(sing_vals) + 1),
        sing_vals,
        marker=".",
        s=50,
        zorder=10,
        color=color,
    )
    plt.scatter(
        elbow_inds, sing_vals[elbow_inds - 1], marker="x", s=50, zorder=10, color=color
    )
    ax.set(ylabel="Singular value", xlabel="Index")
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    return ax


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(
    left_sing_vals, left_elbow_inds, color=network_palette["Left"], ax=ax, label="Left"
)
screeplot(
    right_sing_vals,
    right_elbow_inds,
    color=network_palette["Right"],
    ax=ax,
    label="Right",
)
ax.legend()
stashfig(f"screeplot")

#%% [markdown]
# ## Hypothesis testing on the embeddings
# To test whether the distribution of latent positions is different, we use the approach
# of the "nonpar" test, also called the latent distribution test. Here, for the backend
# 2-sample test, we use distance correlation (Dcorr).
#%%
test = "dcorr"
workers = -1
auto = True
if auto:
    n_bootstraps = None
else:
    n_bootstraps = 500


def run_test(
    X1,
    X2,
    rows=None,
    info={},
    auto=auto,
    n_bootstraps=n_bootstraps,
    workers=workers,
    test=test,
    print_out=False,
):
    currtime = time.time()
    test_obj = KSample(test)
    tstat, pvalue = test_obj.test(
        X1,
        X2,
        reps=n_bootstraps,
        workers=workers,
        auto=auto,
    )
    elapsed = time.time() - currtime
    row = {
        "pvalue": pvalue,
        "tstat": tstat,
        "elapsed": elapsed,
    }
    row.update(info)
    if print_out:
        pprint.pprint(row)
    if rows is not None:
        rows.append(row)
    else:
        return row


#%%

max_n_components = 21
rows = []
for n_components in range(1, max_n_components):
    left_latent, right_latent = embed_and_align(
        ll_adj, rr_adj, n_components=n_components
    )

    run_test(
        left_latent,
        right_latent,
        rows=rows,
        info={"alignment": "SOP", "n_components": n_components},
        print_out=True,
    )
results = pd.DataFrame(rows)

#%% [markdown]
# ### Plot p-values for this version of the test
#%%
def plot_pvalues(results, line_locs=[0.05, 0.005, 0.0005]):
    results = results.copy()

    styles = ["-", "--", ":"]
    line_kws = dict(color="black", alpha=0.7, linewidth=1.5, zorder=-1)

    # plot p-values by embedding dimension
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.scatterplot(
        data=results,
        x="n_components",
        y="pvalue",
        ax=ax,
        s=80,
    )
    ax.set_yscale("log")
    styles = ["-", "--", ":"]
    line_kws = dict(color="black", alpha=0.7, linewidth=1.5, zorder=-1)
    for loc, style in zip(line_locs, styles):
        ax.axhline(loc, linestyle=style, **line_kws)
        ax.text(ax.get_xlim()[-1] + 0.1, loc, loc, ha="left", va="center")
    ax.set(xlabel="Dimension", ylabel="p-value")
    # ax.get_legend().remove()
    # ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", title="Alignment")

    xlim = ax.get_xlim()
    for x in range(1, int(xlim[1]), 2):
        ax.axvspan(x - 0.5, x + 0.5, color="lightgrey", alpha=0.2, linewidth=0)

    ax.set(xticks=np.arange(1, results["n_components"].max(), 2))

    plt.tight_layout()


plot_pvalues(results)

stashfig("p-values-embed-equals-align")

#%% [markdown]
# ## Investigate what happened

#%% [markdown]
# ### Look at the embeddings
#%%


def plot_latents(left, right, title="", n_show=4, lower_triu=True):
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(
        plot_data[:, :n_show], labels=labels, title=title, palette=network_palette
    )
    if lower_triu:
        axs = pg.axes
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                if i <= j:
                    ax = axs[i, j]
                    ax.remove()
        pg._legend.remove()
    return pg


#%% [markdown]
# ### No alignment
#%%
test_n_components = [5]
for n_components in test_n_components:
    left_latent, right_latent = embed_and_align(
        ll_adj, rr_adj, n_components=n_components, align=False
    )
    plot_latents(
        left_latent,
        right_latent,
        n_show=n_components,
        lower_triu=True,
    )
    stashfig(f"embedding-d={n_components}-no-align")

#%% [markdown]
# ### Align in same dimensionality we will use for testing
#%%

test_n_components = [5]
align_n_components = 5
for n_components in test_n_components:
    left_latent, right_latent = embed_and_align(
        ll_adj, rr_adj, n_components=n_components, align_n_components=align_n_components
    )
    plot_latents(
        left_latent,
        right_latent,
        n_show=n_components,
        lower_triu=True,
    )
    stashfig(f"embedding-d={n_components}-align={align_n_components}")

#%% [markdown]
# ### Illustration

#%%
n_components = 8
align_n_components = 8
left_latent, right_latent = embed_and_align(
    ll_adj, rr_adj, n_components=n_components, align=False
)
plot_latents(
    left_latent,
    right_latent,
    n_show=n_components,
)
stashfig(f"embedding-d={n_components}-no-align")

left_latent, right_latent = embed_and_align(
    ll_adj, rr_adj, n_components=n_components, align_n_components=align_n_components
)
plot_latents(
    left_latent,
    right_latent,
    n_show=n_components,
)
stashfig(f"embedding-d={n_components}-align={align_n_components}")

#%%
align_n_components = 8
for n_components in test_n_components:
    left_latent, right_latent = embed_and_align(
        ll_adj, rr_adj, n_components=n_components, align_n_components=align_n_components
    )
    plot_latents(
        left_latent,
        right_latent,
        n_show=n_components,
    )
    stashfig(f"embedding-d={n_components}-align={align_n_components}")

#%%

max_n_components = 21
rows = []
for n_components in range(1, max_n_components):
    left_latent, right_latent = embed_and_align(
        ll_adj, rr_adj, n_components=n_components, align_n_components=max_n_components
    )

    run_test(
        left_latent,
        right_latent,
        rows=rows,
        info={"alignment": "SOP", "n_components": n_components},
        print_out=True,
    )
results = pd.DataFrame(rows)

plot_pvalues(results)

stashfig(f"p-values-align={max_n_components}")

#%% [markdown]
# ### Run the two-sample test for varying embedding dimension
#%%


rows = []
for n_components in np.arange(1, align_n_components + 1):
    left_composite_latent = np.concatenate(
        (X_ll[:, :n_components], Y_ll[:, :n_components]), axis=1
    )
    right_composite_latent = np.concatenate(
        (X_rr[:, :n_components], Y_rr[:, :n_components]), axis=1
    )

    run_test(
        left_composite_latent,
        right_composite_latent,
        rows,
        info={"alignment": "SOP", "n_components": n_components},
    )

results = pd.DataFrame(rows)


#%% [markdown]
# ### Plot the 2-sample test p-values by varying dimension
# Note: these are on a log y-scale.
#%%


def plot_pvalues(results, line_locs=[0.05, 0.005, 0.0005]):
    results = results.copy()

    styles = ["-", "--", ":"]
    line_kws = dict(color="black", alpha=0.7, linewidth=1.5, zorder=-1)

    # plot p-values by embedding dimension
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.scatterplot(
        data=results,
        x="n_components",
        y="pvalue",
        ax=ax,
        s=40,
    )
    ax.set_yscale("log")
    styles = ["-", "--", ":"]
    line_kws = dict(color="black", alpha=0.7, linewidth=1.5, zorder=-1)
    for loc, style in zip(line_locs, styles):
        ax.axhline(loc, linestyle=style, **line_kws)
        ax.text(ax.get_xlim()[-1] + 0.1, loc, loc, ha="left", va="center")
    ax.set(xlabel="Dimension", ylabel="p-value")

    xlim = ax.get_xlim()
    for x in range(1, int(xlim[1]), 2):
        ax.axvspan(x - 0.5, x + 0.5, color="lightgrey", alpha=0.2, linewidth=0)

    plt.tight_layout()


plot_pvalues(results)
stashfig(
    f"naive-pvalues-test={test}-n_bootstraps={n_bootstraps}-preprocess={preprocess}"
)

#%% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
