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

FILENAME = "plot_perturbations_unmatched"


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

result_path = Path("bilateral-connectome/results/outputs/perturbations_unmatched")

# reopen
simple_results = pd.read_csv(result_path / "unmatched_power_simple.csv", index_col=0)

with open(result_path / "unmatched_power_full.pickle", "rb") as f:
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

perturbations = results["perturbation"].unique()

grid = sns.FacetGrid(
    results,
    col="perturbation",
    col_wrap=min(4, len(perturbations)),
    sharex=False,
    sharey=True,
    hue="test",
    height=4,
)
grid.map_dataframe(sns.lineplot, x="effect_size", y="pvalue")
grid.add_legend(title="Test")
grid.set_ylabels("p-value")
grid.set_xlabels("Effect size")
grid.set_titles("{col_name}")
grid.set(ylim=(-0.01, 1.01))
gluefig("pvalue-grid", grid.figure)

#%%
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
grid.set_ylabels("Empirical power")
grid.set_xlabels("Effect size")
# grid.set_titles("{col_name}")
gluefig("pvalue-grid", grid.figure)
