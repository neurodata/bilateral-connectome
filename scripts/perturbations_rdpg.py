#%%
from pkg.utils import set_warnings

set_warnings()

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_random_seed
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.perturb import (
    add_edges,
    add_edges_subgraph,
    remove_edges,
    remove_edges_subgraph,
    shuffle_edges,
    shuffle_edges_subgraph,
)
from pkg.plot import set_theme
from pkg.stats import degree_test, erdos_renyi_test, rdpg_test, stochastic_block_test
from pkg.utils import get_seeds
from tqdm import tqdm


DISPLAY_FIGS = True

FILENAME = "perturbations_rdpg"


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

left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values


left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))

seeds = get_seeds(left_nodes, right_nodes)

#%%
random_state = np.random.default_rng(8888)
adj = right_adj
nodes = right_nodes
labels1 = right_labels
labels2 = right_labels
seeds = get_seeds(right_nodes, right_nodes)

#%%

groups = np.unique(labels1)

#%%


def swap_incident_edges_subgraph(adjacency, nodes, effect_size=8, random_seed=None):
    if isinstance(random_seed, np.integer) or random_seed is None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    adjacency = adjacency.copy()

    # choose some number of nodes to mess with
    perturb_index = rng.choice(nodes, size=effect_size, replace=False)

    # take each node, and randomly swap the edges incident to that node
    for i in perturb_index:
        perm_inds = np.random.permutation(len(adjacency))
        adjacency[i] = adjacency[i, perm_inds]
        perm_inds = np.random.permutation(len(adjacency))
        adjacency[:, i] = adjacency[perm_inds, i]

    return adjacency


rows = []
n_sims = 5
for group in groups[:3]:
    print(group)
    group_nodes = nodes[nodes["simple_group"] == group]["inds"]
    max_effect_size = len(group_nodes)
    # for effect_size in np.linspace(0.1 * max_effect_size, max_effect_size, 10):
    effect_steps = np.geomspace(
        start=0.1 * max_effect_size, stop=max_effect_size, dtype=int, num=5
    )
    print(effect_steps)
    for effect_size in effect_steps:
        for sim in range(n_sims):
            perturb_adj = swap_incident_edges_subgraph(
                adj, group_nodes, effect_size=effect_size, random_seed=rng
            )
            stat, pvalue, misc = rdpg_test(
                adj,
                perturb_adj,
                seeds=seeds,
                normalize_nodes=True,
                align_n_components=16,
                n_components=4,
            )
            row = {
                "stat": stat,
                "pvalue": pvalue,
                "sim": sim,
                "effect_size": effect_size,
                "group": group,
                **misc,
            }
            rows.append(row)

#%%

data = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=data, x="effect_size", hue="group", y="pvalue", palette=node_palette)
ax.get_legend().set_title("Cell type")
ax.set(yscale="log", xlabel="# of perturbed neurons", ylabel="p-value")
