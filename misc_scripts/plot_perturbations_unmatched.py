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
results = pd.read_csv(result_path / "unmatched_power_simple.csv", index_col=0)

results["model_postfix"] = results["model_postfix"].fillna("")
# with open(result_path / "unmatched_power_full.pickle", "rb") as f:
#     results = pickle.load(f)

# results.columns

#%%
# filter
results = results[results["test"] != "Degree"].copy()

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

# results["pvalue"] = results["pvalue"].map(lambda x: max(x, 1e-8))

blues = sns.color_palette("Blues", 8)
oranges = sns.color_palette("Oranges", 8)
greens = sns.color_palette("Greens", 8)
palette = {
    "ER": blues[-1],
    "SBM-f": oranges[-3],
    "SBM-m": oranges[-4],
    "RDPG": greens[-1],
    "RDPG-n": greens[-4],
}
dashes = {"": "", "f": "", "m": (1, 1), "n": (2, 1)}


set_theme(font_scale=1.25)
grid = sns.FacetGrid(
    results,
    col="target",
    row="perturbation_type",
    sharex="col",
    sharey=True,
    hue="test",
    height=4,
    palette=palette,
)

grid.map_dataframe(
    sns.lineplot, x="effect_size", y="pvalue", style="model_postfix", dashes=dashes
)
grid.add_legend(
    title="Test",
    ncol=results["test"].nunique(),
    loc="lower left",
    bbox_to_anchor=(0.05, 0.96),
    title_fontsize="x-large",
)
leg = grid._legend
leg._legend_box.align = "left"
title = leg.get_title()
title.set_fontsize("large")

grid.set_ylabels("p-value")
grid.set_xlabels("# of edges")
grid.set_titles("{col_name}")
grid.set(ylim=(-0.01, 1.01))
grid.figure.text(-0.03, 0.70, "Remove\nedges", fontsize="x-large")
grid.figure.text(-0.03, 0.25, "Shuffle\nedges", fontsize="x-large")

mean_results = (
    results.groupby(["test", "perturbation", "effect_size"]).mean().reset_index()
)
detected_results = mean_results[mean_results["pvalue"] < 0.05]
first_detected_results = detected_results.groupby(["test", "perturbation"]).first()

# for (perturbation_type, target), ax in grid._axes_dict.items():
#     perturbation = perturbation_type + " edges (" + target + ")"
#     for test in results["test"].unique():
#         try:
#             effect_size = first_detected_results.loc[
#                 (test, perturbation), "effect_size"
#             ]
#             ax.plot([effect_size, effect_size], [0, 1], color=palette[test])
#         except KeyError:
#             pass

axs = grid.axes
for ax in axs[1, :]:
    ax.set_title("")

gluefig("pvalue-grid", grid.figure)

#%%


grid = sns.FacetGrid(
    first_detected_results,
    col="target",
    row="perturbation",
    sharex=False,
    sharey=False,
    hue="test",
    height=4,
    palette=palette,
)

grid.map_dataframe(
    sns.barplot,
    x="test",
    y="effect_size",  # style="model_postfix", dashes=dashes
)

#%%
# grid = sns.FacetGrid(
#     results,
#     col="target",
#     row="perturbation_type",
#     sharex="col",
#     sharey=True,
#     hue="test",
#     height=4,
# )
# grid.map_dataframe(sns.lineplot, x="effect_size", y="pvalue")
# grid.add_legend(
#     title="Test",
#     ncol=results["test"].nunique(),
#     loc="lower left",
#     bbox_to_anchor=(0.05, 0.95),
# )
# grid.set_ylabels("p-value")
# grid.set_xlabels("Effect size")
# grid.set_titles("{col_name}")
# grid.set(ylim=(-0.01, 1.01))
# gluefig("pvalue-grid", grid.figure)


# #%%
# grid = sns.FacetGrid(
#     results,
#     col="perturbation",
#     col_wrap=min(3, len(perturbations)),
#     sharex=False,
#     sharey=False,
#     hue="test",
#     height=6,
# )
# grid.map_dataframe(sns.lineplot, x="effect_size", y="power_indicator")
# grid.add_legend(title="Test")
# grid.set_ylabels("Empirical power")
# grid.set_xlabels("Effect size")
# # grid.set_titles("{col_name}")
# gluefig("pvalue-grid", grid.figure)
