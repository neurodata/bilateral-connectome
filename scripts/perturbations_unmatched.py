#%% [markdown]
# ## Preliminaries
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
from pkg.data import (
    load_network_palette,
    load_node_palette,
    load_unmatched,
)
from pkg.io import savefig
from pkg.perturb import add_edges, remove_edges, shuffle_edges
from pkg.plot import set_theme
from pkg.stats import erdos_renyi_test, rdpg_test, stochastic_block_test
from pkg.utils import get_seeds

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
labels1 = right_labels
labels2 = right_labels
n_sims = 1
effect_sizes = [0, 20, 40, 60, 80]
seeds = (seeds[1], seeds[1])

n_components = 8

rows = []

tests = {"ER": erdos_renyi_test, "SBM": stochastic_block_test, "RDPG": rdpg_test}
test_options = {
    "ER": [{}],
    "SBM": [{"labels1": labels1, "labels2": labels2}],
    "RDPG": [{"n_components": n_components, "seeds": seeds, "normalize_nodes": False}],
}
perturbations = {
    "Remove edges (global)": remove_edges,
    "Add edges (global)": add_edges,
    "Shuffle edges (global)": shuffle_edges,
}

for perturbation_name, perturb in perturbations.items():
    for effect_size in effect_sizes:
        for sim in range(n_sims):
            seed = get_random_seed(random_state)
            perturb_adj = perturb(adj, effect_size=effect_size, random_seed=seed)
            for test_name, test in tests.items():
                option_sets = test_options[test_name]
                for options in option_sets:
                    stat, pvalue, other = test(adj, perturb_adj, **options)
                    row = {
                        "stat": stat,
                        "pvalue": pvalue,
                        "other": other,
                        "test": test_name,
                        "perturbation": perturbation_name,
                        "effect_size": effect_size,
                        "sim": sim,
                        **options,
                    }
                    rows.append(row)

results = pd.DataFrame(rows)
results

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.scatterplot(data=results, x="effect_size", y="pvalue", hue="test", ax=ax)
