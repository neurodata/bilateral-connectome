#%% [markdown]
# ## Preliminaries
#%%

from os import uname
import time
from giskard.plot.old_matrixplot import remove_shared_ax

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_random_seed
from graspologic.utils import binarize
from pkg.data import (
    load_maggot_graph,
    load_network_palette,
    load_node_palette,
    select_nice_nodes,
)
from pkg.io import savefig
from pkg.perturb import add_edges, remove_edges, shuffle_edges
from pkg.plot import set_theme
from pkg.stats import erdos_renyi_test, stochastic_block_test
from seaborn.utils import relative_luminance


def stashfig(name, **kwargs):
    foldername = "main_loop"
    savefig(name, foldername=foldername, **kwargs)


# %% [markdown]
# ## Load and process data
#%%

t0 = time.time()
set_theme()

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

#%%

GROUP_KEY = "simple_group"

mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True)
left_nodes = left_mg.nodes
right_nodes = right_mg.nodes

left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj
left_adj = binarize(left_adj)
right_adj = binarize(right_adj)

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values


#%%
random_state = np.random.default_rng(8888)
adj = right_adj
labels1 = right_labels
labels2 = right_labels
n_sims = 1
effect_sizes = [0, 20, 40, 60, 80]
rows = []

tests = {"ER": erdos_renyi_test, "SBM": stochastic_block_test}
test_options = {
    "ER": [{"method": "agresti-caffo"}],
    "SBM": [{"labels1": labels1, "labels2": labels2, "method": "agresti-caffo"}],
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

#%%
