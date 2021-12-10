#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import set_warnings

set_warnings()

import pickle
import time
from pathlib import Path
from pprint import pprint

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

DISPLAY_FIGS = False

FILENAME = "perturbations_unmatched"


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
n_sims = 1
# effect_sizes = [256, 512, 2048, 4096, 8192]
effect_sizes = np.linspace(0, 2000, 11).astype(int)
seeds = (seeds[1], seeds[1])

n_components = 8

#%%

KCs_nodes = nodes[nodes["simple_group"] == "KCs"]["inds"]


def remove_edges_KCs_KCs(adjacency, **kwargs):
    return remove_edges_subgraph(adjacency, KCs_nodes, KCs_nodes, **kwargs)


#%%

rows = []

tests = {
    "ER": erdos_renyi_test,
    "SBM": stochastic_block_test,
    "SBM-m": stochastic_block_test,
    "Degree": degree_test,
    "RDPG": rdpg_test,
    "RDPG-n": rdpg_test,
}
test_options = {
    "ER": {},
    "SBM": {"labels1": labels1, "labels2": labels2, "combine_method": "fisher"},
    "SBM-m": {"labels1": labels1, "labels2": labels2, "combine_method": "min"},
    "Degree": {},
    "RDPG": {"n_components": n_components, "seeds": seeds, "normalize_nodes": False},
    "RDPG-n": {"n_components": n_components, "seeds": seeds, "normalize_nodes": True},
}
perturbations = {
    "Remove edges (global)": remove_edges,
    r"Remove edges (KCs $\rightarrow$ KCs)": remove_edges_KCs_KCs,
    "Shuffle edges (global)": shuffle_edges,
    # "Add edges (global)": add_edges,
}

n_runs = len(tests) * n_sims * len(effect_sizes)

from joblib import Parallel, delayed

def perturb_and_run_tests()

for perturbation_name, perturb in perturbations.items():
    for effect_size in effect_sizes:
        for sim in range(n_sims):
            currtime = time.time()
            seed = get_random_seed(random_state)
            perturb_adj = perturb(adj, effect_size=effect_size, random_seed=seed)
            perturb_elapsed = time.time() - currtime

            for test_name, test in tests.items():
                options = test_options[test_name]

                currtime = time.time()
                stat, pvalue, other = test(adj, perturb_adj, **options)
                test_elapsed = time.time() - currtime

                row = {
                    "stat": stat,
                    "pvalue": pvalue,
                    "other": other,
                    "test": test_name,
                    "perturbation": perturbation_name,
                    "effect_size": effect_size,
                    "sim": sim,
                    "perturb_elapsed": perturb_elapsed,
                    "test_elapsed": test_elapsed,
                    **options,
                }
                rows.append(row)

                print(f"{perturbation_name}-{effect_size}-{sim}-{test_name}")

results = pd.DataFrame(rows)

#%%

out_path = Path("bilateral-connectome/results/outputs")

# save
simple_results = results[
    [
        "stat",
        "pvalue",
        "test",
        "perturbation",
        "effect_size",
        "sim",
        "perturb_elapsed",
        "test_elapsed",
    ]
]
simple_results.to_csv(out_path / "unmatched_power_simple.csv")

with open(out_path / "unmatched_power_full.pickle", "wb") as f:
    pickle.dump(results, f)

# reopen
simple_results = pd.read_csv(out_path / "unmatched_power_simple.csv", index_col=0)

with open(out_path / "unmatched_power_full.pickle", "rb") as f:
    results = pickle.load(f)

#%%


def check_power(pvalues, alpha=0.05):
    n_significant = (pvalues <= alpha).sum()
    power = (n_significant) / (len(pvalues))
    return power


power_results = (
    results.groupby(["test", "perturbation", "effect_size"]).mean().reset_index()
)

power = (
    results.groupby(["test", "perturbation", "effect_size"])["pvalue"]
    .agg(check_power)
    .reset_index()
)
power.rename(columns=dict(pvalue="power"), inplace=True)
power_results["power"] = power["power"]
results["power_indicator"] = (results["pvalue"] < 0.05).astype(float)
results["power_indicator"] = results["power_indicator"] + np.random.normal(
    0, 0.01, size=len(results)
)

# %%
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# sns.scatterplot(data=results, x="effect_size", y="pvalue", hue="test", ax=ax)

grid = sns.FacetGrid(
    results,
    col="perturbation",
    col_wrap=min(3, len(perturbations)),
    sharex=False,
    sharey=False,
    hue="test",
    height=6,
)
grid.map_dataframe(sns.lineplot, x="effect_size", y="pvalue")
grid.add_legend(title="Test")
grid.set_ylabels("Empirical power")
grid.set_xlabels("Effect size")
grid.set_titles("{col_name}")
gluefig("power-grid", grid.figure)

#%%


# # source_nodes = right_nodes[right_nodes["simple_group"] == "KCs"]["inds"]
# # # subgraph = adj[source_nodes][:, source_nodes]
# # # remove_edges(subgraph, effect_size=100)
# # perturb_subgraph(adj, remove_edges, source_nodes, source_nodes)

# #%%

# KCs_nodes = nodes[nodes["simple_group"] == "KCs"]["inds"]


# def remove_edges_KCs_KCs(adjacency, **kwargs):
#     return remove_edges_subgraph(adjacency, KCs_nodes, KCs_nodes, **kwargs)


# new_adj = remove_edges_KCs_KCs(adj, effect_size=100)

# #%%

# A = np.zeros((3,3))
# A[np.ix_([1,2], [1,2])] = np.ones((2,2))
# A
# # %%
