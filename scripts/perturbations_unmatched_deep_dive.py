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
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.perturb import (
    add_edges,
    remove_edges,
    shuffle_edges,
    add_edges_subgraph,
    remove_edges_subgraph,
    shuffle_edges_subgraph,
)
from pkg.plot import set_theme
from pkg.stats import degree_test, erdos_renyi_test, rdpg_test, stochastic_block_test
from pkg.utils import get_seeds
from tqdm import tqdm

DISPLAY_FIGS = True

FILENAME = "perturbations_unmatched_deep_dive"


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
effect_sizes = np.linspace(0, 3000, 30).astype(int)
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
    "Degree": degree_test,
    # "RDPG": rdpg_test,
    # "RDPG-n":rdpg_test,
}
test_options = {
    "ER": [{}],
    "SBM": [{"labels1": labels1, "labels2": labels2, "combine_method": "min"}],
    "Degree": [{}],
    # "RDPG": [{"n_components": n_components, "seeds": seeds, "normalize_nodes": False}],
    # "RDPG-n": [{"n_components": n_components, "seeds": seeds, "normalize_nodes": True}],
}
perturbations = {
    "Remove edges (global)": remove_edges,
    r"Remove edges (KCs $\rightarrow$ KCs)": remove_edges_KCs_KCs
    # "Add edges (global)": add_edges,
    # "Shuffle edges (global)": shuffle_edges,
}

n_runs = len(tests) * n_sims * len(effect_sizes)

for perturbation_name, perturb in perturbations.items():
    for effect_size in tqdm(effect_sizes):
        for sim in range(n_sims):
            currtime = time.time()
            seed = get_random_seed(random_state)
            perturb_adj = perturb(adj, effect_size=effect_size, random_seed=seed)
            perturb_elapsed = time.time() - currtime

            for test_name, test in tests.items():
                option_sets = test_options[test_name]
                for options in option_sets:
                    currtime = time.time()
                    stat, pvalue, other = test(adj, perturb_adj, **options)
                    test_elapsed = time.time() - currtime
                    if test_name == "SBM":
                        uncorrected_pvalues = other["uncorrected_pvalues"]
                        other["KCs_pvalues"] = uncorrected_pvalues.loc["KCs", "KCs"]
                    row = {
                        "stat": stat,
                        "pvalue": pvalue,
                        "test": test_name,
                        "perturbation": perturbation_name,
                        "effect_size": effect_size,
                        "sim": sim,
                        "perturb_elapsed": perturb_elapsed,
                        "test_elapsed": test_elapsed,
                        **options,
                        **other,
                    }
                    rows.append(row)

results = pd.DataFrame(rows)

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
    0, 0.0025, size=len(results)
)
# %%
grid = sns.FacetGrid(
    results,
    col="perturbation",
    col_wrap=min(3, len(perturbations)),
    sharex=False,
    sharey=False,
    hue="test",
    height=6,
)
grid.map_dataframe(sns.lineplot, x="effect_size", y="power_indicator")
grid.add_legend(title="Test")
grid.set_ylabels(r"Empirical power ($\alpha = 0.05$)")
grid.set_xlabels("Effect size")
grid.set_titles("{col_name}")
gluefig("power", grid.figure)
# %%
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
grid.set_ylabels(r"p-value")
grid.set_xlabels("Effect size")
grid.set_titles("{col_name}")
gluefig("pvalues", grid.figure)
#%%
subresults = results[results["perturbation"] == r"Remove edges (KCs $\rightarrow$ KCs)"]
subresults = subresults[subresults["test"] == "SBM"].copy()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=subresults,
    x="effect_size",
    y="KCs_pvalues",
    ax=ax,
    label=r"KCs $\rightarrow$ KCs",
)

mean_pvalues = []
all_pvalues = []
for i in range(len(subresults)):
    row = subresults.iloc[i]
    vals = row["uncorrected_pvalues"].values
    mean = np.nanmean(vals)
    mean_pvalues.append(mean)
    for j, pvalue in enumerate(vals.ravel()):
        all_pvalues.append(
            {"effect_size": row["effect_size"], "pvalue": pvalue, "j": j}
        )

all_pvalues = pd.DataFrame(all_pvalues)
subresults["mean_pvalues"] = mean_pvalues

sns.lineplot(
    data=subresults, x="effect_size", y="mean_pvalues", ax=ax, label="Mean p-value"
)

ax.set(ylabel="p-value", xlabel="Effect size (# edges removed)")

sns.lineplot(data=subresults, x="effect_size", y="pvalue", label="Fisher's combined")

ax.set_title(r"Remove edges (KCs $\rightarrow$ KCs)")

gluefig("split_pvalues", fig)
