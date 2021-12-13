#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import set_warnings

set_warnings()

import datetime
import pickle
import time
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_random_seed
from joblib import Parallel, delayed
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
import re

DISPLAY_FIGS = True

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
n_sims = 10
# effect_sizes = [256, 512, 2048, 4096, 8192]
effect_sizes = np.linspace(0, 3000, 16).astype(int)
seeds = (seeds[1], seeds[1])

n_components = 8

#%%

KCs_nodes = nodes[nodes["simple_group"] == "KCs"]["inds"]


def remove_edges_KCs_KCs(adjacency, **kwargs):
    return remove_edges_subgraph(adjacency, KCs_nodes, KCs_nodes, **kwargs)


LHNs_nodes = nodes[nodes["simple_group"] == "LHNs"]["inds"]


def remove_edges_LHNs_LHNs(adjacency, **kwargs):
    return remove_edges_subgraph(adjacency, LHNs_nodes, LHNs_nodes, **kwargs)


PNs_nodes = nodes[nodes["simple_group"] == "PNs"]["inds"]


def remove_edges_PNs_LHNs(adjacency, **kwargs):
    return remove_edges_subgraph(adjacency, PNs_nodes, LHNs_nodes, **kwargs)


# shuffling
def shuffle_edges_KCs_KCs(adjacency, **kwargs):
    return shuffle_edges_subgraph(adjacency, KCs_nodes, KCs_nodes, **kwargs)


def shuffle_edges_LHNs_LHNs(adjacency, **kwargs):
    return shuffle_edges_subgraph(adjacency, LHNs_nodes, LHNs_nodes, **kwargs)


def shuffle_edges_PNs_LHNs(adjacency, **kwargs):
    return shuffle_edges_subgraph(adjacency, PNs_nodes, LHNs_nodes, **kwargs)


#%%

nodes["simple_group"].value_counts()

#%%

rows = []

tests = {
    "ER": erdos_renyi_test,
    "SBM-f": stochastic_block_test,
    "SBM-m": stochastic_block_test,
    "Degree": degree_test,
    "RDPG": rdpg_test,
    "RDPG-n": rdpg_test,
}
test_options = {
    "ER": {},
    "SBM-f": {"labels1": labels1, "labels2": labels2, "combine_method": "fisher"},
    "SBM-m": {"labels1": labels1, "labels2": labels2, "combine_method": "min"},
    "Degree": {},
    "RDPG": {"n_components": n_components, "seeds": seeds, "normalize_nodes": False},
    "RDPG-n": {"n_components": n_components, "seeds": seeds, "normalize_nodes": True},
}
perturbations = {
    "Remove edges (global)": remove_edges,
    r"Remove edges (KCs$\rightarrow$KCs)": remove_edges_KCs_KCs,
    r"Remove edges (LHNs$\rightarrow$LHNs)": remove_edges_LHNs_LHNs,
    r"Remove edges (PNs$\rightarrow$LHNs)": remove_edges_PNs_LHNs,
    "Shuffle edges (global)": shuffle_edges,
    r"Shuffle edges (KCs$\rightarrow$KCs)": shuffle_edges_KCs_KCs,
    r"Shuffle edges (LHNs$\rightarrow$LHNs)": shuffle_edges_LHNs_LHNs,
    r"Shuffle edges (PNs$\rightarrow$LHNs)": shuffle_edges_PNs_LHNs,
}

n_runs = len(tests) * n_sims * len(effect_sizes)


def perturb_and_run_tests(seed, perturbation_name, perturb, effect_size, sim):
    currtime = time.time()
    perturb_adj = perturb(adj, effect_size=effect_size, random_seed=seed)
    perturb_elapsed = time.time() - currtime

    target = re.search("\((.*?)\)", perturbation_name).group(1)
    perturbation_type = perturbation_name.split(" ")[0]

    if perturb_adj is None:
        return []

    rows = []
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
            "target": target.capitalize(),
            "perturbation_type": perturbation_type,
            **options,
        }
        rows.append(row)

    return rows


parameters = []
for perturbation_name, perturb in perturbations.items():
    for effect_size in effect_sizes:
        for sim in range(n_sims):
            seed = get_random_seed(random_state)
            params = {
                "seed": seed,
                "perturbation_name": perturbation_name,
                "perturb": perturb,
                "effect_size": effect_size,
                "sim": sim,
            }
            parameters.append(params)

overall_time = time.time()
parallel = Parallel(n_jobs=-2, verbose=10)
chunks_of_rows = parallel(
    delayed(perturb_and_run_tests)(**params) for params in parameters
)
overall_elapsed = time.time() - overall_time
delta = datetime.timedelta(seconds=overall_elapsed)

time.sleep(1)
print()
print(f"Entire set of power simulations took {delta}")
print()

rows = []
for chunk in chunks_of_rows:
    rows += chunk

results = pd.DataFrame(rows)

#%%

out_path = Path("bilateral-connectome/results/outputs/perturbations_unmatched")

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
