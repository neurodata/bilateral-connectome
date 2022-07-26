#%%
import datetime
from re import sub
import time

import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import merge_axes, soft_axis_off
from graspologic.simulations import er_np
from matplotlib.collections import LineCollection
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import SmartSVG, networkplot_simple, set_theme
from pkg.plot.er import plot_density
from pkg.stats import erdos_renyi_test
from pkg.utils import sample_toy_networks
from svgutils.compose import Figure, Panel, Text
from pkg.plot import draw_hypothesis_box


DISPLAY_FIGS = True

FILENAME = "matched_vs_unmatched_sims"


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


t0 = time.time()
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

#%%
from graspologic.simulations import er_np
from pkg.perturb import shuffle_edges, add_edges
from pkg.stats import erdos_renyi_test, erdos_renyi_test_paired
from tqdm.autonotebook import tqdm

p = 0.1
n = 50

effect_sizes = np.linspace(0, 50, 51, dtype=int)
n_sims = 10
rows = []

with tqdm(total=n_sims * len(effect_sizes) * 2) as pbar:
    for effect_size in effect_sizes:
        for sim in range(n_sims):
            A1 = er_np(n, p, directed=True)
            A2 = add_edges(A1, effect_size=effect_size)

            for test in ["er", "er_paired"]:
                if test == "er":
                    name = "Density test"
                    stat, pvalue, misc = erdos_renyi_test(A1, A2)
                elif test == "er_paired":
                    name = "Paired density test"
                    stat, pvalue, misc = erdos_renyi_test_paired(A1, A2)
                else:
                    raise ValueError()

                result = {
                    "test": test,
                    "stat": stat,
                    "pvalue": pvalue,
                    "effect_size": effect_size,
                    "name": name,
                }
                rows.append(result)
                pbar.update(1)
#%%
results = pd.DataFrame(rows)

#%%


def add_alpha_line(ax, xytext=(-45, -15)):
    ax.axhline(0.05, color="black", linestyle=":", zorder=-1)
    ax.annotate(
        r"0.05",
        (ax.get_xlim()[0], 0.05),
        xytext=xytext,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="black"),
        clip_on=False,
        ha="right",
    )


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors = sns.color_palette()
palette = dict(zip(["er", "er_paired"], colors))
sns.lineplot(data=results, x="effect_size", y="pvalue", hue="name")
ax.set_ylabel("p-value") 
ax.set_xlabel("# added edges (effect size)", labelpad=15)
sns.move_legend(ax, loc="upper right", title="")
add_alpha_line(ax, xytext=(-45, 15))

ax.tick_params(axis='both', length=5)
plt.autoscale(False)

xytexts=[(10,-35), (-10,-35)]
for i, test in enumerate(["er", "er_paired"]):
    sub_results = results[results["test"] == test]
    means = sub_results.groupby("effect_size")["pvalue"].mean()
    x = means[means < 0.05].index[0]
    # x = sub_results[sub_results["pvalue"] < 0.05].iloc[0]["effect_size"]

    ax.axvline(x, ymax=0.095, color=palette[test], linestyle=':', linewidth=3)
    # ax.text(x, 0, x, va='top', ha='center', clip_on=False)
    ax.annotate(
        x,
        (x, ax.get_ylim()[0]),
        xytext=xytexts[i],
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color=palette[test]),
        clip_on=False,
        ha="center",
        color=palette[test]
    )

gluefig("er_power_comparison", fig)

#%%

rng = np.random.default_rng()
n_swap = 10


p = 0.2
n = 50

effect_sizes = np.arange(0, 55, 5)  # np.linspace(0, 50, 51, dtype=int)
n_sims = 1
n_swaps_range = np.arange(0, 55, 5)  # np.linspace(0, 50, 51, dtype=int)
rows = []

with tqdm(total=n_sims * len(effect_sizes) * 2 * len(n_swaps_range)) as pbar:
    for effect_size in effect_sizes:
        for n_swaps in n_swaps_range:
            for sim in range(n_sims):
                A1 = er_np(n, p, directed=True)
                A2 = add_edges(A1, effect_size=effect_size)
                n_keep = n - n_swaps
                keep_inds = np.arange(n_keep)
                swap_inds = np.arange(n_keep, n)
                swap_inds = rng.permutation(swap_inds)
                permutation = np.concatenate((keep_inds, swap_inds))

                A2 = A2[permutation][:, permutation]

                for test in ["er", "er_paired"]:
                    if test == "er":
                        name = "Density test"
                        stat, pvalue, misc = erdos_renyi_test(A1, A2, method="fisher")
                    elif test == "er_paired":
                        name = "Paired density test"
                        stat, pvalue, misc = erdos_renyi_test_paired(A1, A2)
                    else:
                        raise ValueError()

                    result = {
                        "test": test,
                        "stat": stat,
                        "pvalue": pvalue,
                        "effect_size": effect_size,
                        "name": name,
                        "n_swaps": n_swaps,
                    }
                    rows.append(result)
                    pbar.update(1)

results = pd.DataFrame(rows)
results


#%%
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
ax = axs[0]
unpaired_results = results[results["test"] == "er"]
unpaired_square_results = unpaired_results.pivot(
    index="n_swaps", columns="effect_size", values="pvalue"
)
unpaired_square_results = unpaired_square_results.iloc[::-1]
sns.heatmap(unpaired_square_results, ax=ax, square=True, cbar_kws=dict(shrink=0.6))

ax = axs[1]
paired_results = results[results["test"] == "er_paired"]
paired_square_results = paired_results.pivot(
    index="n_swaps", columns="effect_size", values="pvalue"
)
paired_square_results = paired_square_results.iloc[::-1]
sns.heatmap(paired_square_results, ax=ax, square=True, cbar_kws=dict(shrink=0.6))

ax = axs[2]
relative_pvalue = paired_square_results / unpaired_square_results
sns.heatmap(
    relative_pvalue,
    ax=ax,
    square=True,
    cbar_kws=dict(shrink=0.6),
    cmap="RdBu_r",
    center=1,
)
gluefig("pvalue_comparison", fig)

# %%
rng = np.random.default_rng()

n = 100000
p1 = 0.1
p2 = 0.11
x1s = rng.binomial(1, p1, size=n)
x2s = rng.binomial(1, p2, size=n)
x = np.column_stack((x1s, x2s))

chi2_table = np.empty((2, 2))
chi2_table[0, 0] = np.sum(x1s)
chi2_table[0, 1] = n - chi2_table[0, 0]
chi2_table[1, 0] = np.sum(x2s)
chi2_table[1, 1] = n - chi2_table[1, 0]


from scipy.stats import chi2_contingency

from statsmodels.stats.contingency_tables import mcnemar

stat, pvalue, _, _ = chi2_contingency(chi2_table, correction=False)
print(stat)
print(pvalue)


mcnemars_table = np.empty((2, 2))
both_fail = (x.sum(axis=1) == 0).sum()
both_success = (x.sum(axis=1) == 2).sum()
only_x1 = ((x[:, 0] == 1) & (x[:, 1] == 0)).sum()
only_x2 = ((x[:, 0] == 0) & (x[:, 1] == 1)).sum()
mcnemars_table = np.array([[both_success, only_x1], [only_x2, both_fail]])

out = mcnemar(mcnemars_table, exact=False, correction=False)
stat = out.statistic
pvalue = out.pvalue
print(stat)
print(pvalue)
