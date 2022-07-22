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

sns.lineplot(data=results, x="effect_size", y="pvalue", hue="name")
ax.set(ylabel="p-value", xlabel="# perturbed edges")
sns.move_legend(ax, loc="upper right", title="")
add_alpha_line(ax, xytext=(-45, 15))
gluefig("er-power-comparison", fig)

# ax.axhline(0.05, color="black", linestyle="--")
# ax.text(max(effect_sizes), 0.05, "alpha")
# ax.set_yscale('log')

#%%

# #%%

# from graspologic.simulations import er_np
# from pkg.perturb import remove_edges
# from pkg.stats import compute_density, erdos_renyi_test, erdos_renyi_test_paired
# from tqdm import tqdm

# p1 = 0.1
# n = 20
# n_sims_alt = 100
# n_sims_null = 1000
# p_equalize = 0.5
# ps = np.linspace(0.1, 0.3, 10)

# rows = []
# for p2 in ps:
#     if p2 == p1:
#         n_sims = n_sims_null
#     else:
#         n_sims = n_sims_alt
#     for sim in tqdm(range(n_sims)):
#         A1 = er_np(n, p1, directed=True)
#         A2 = er_np(n, p2, directed=True)

#         n_set = A1.size * p_equalize
#         choice_edge_indices = np.random.choice(A1.size, size=n_set, replace=False)
#         row_inds, col_inds = np.unravel_index(choice_edge_indices, A1.shape)
#         A2[]

#         # density_before = compute_density(A2)
#         # flat_edges1 = np.nonzero(A1.ravel())[0]
#         # flat_edges2 = np.nonzero(A2.ravel())[0]
#         # edges1_not2 = np.setdiff1d(flat_edges1, flat_edges2)
#         # n_edges = np.count_nonzero(A1)
#         # n_set = int(n_edges * p_equalize)
#         # A2 = remove_edges(A2, effect_size=n_set)
#         #
#         # A2[row_inds, col_inds] = 1
#         # density_after = compute_density(A2)

#         stat, pvalue, misc = erdos_renyi_test(A1, A2)
#         rows.append(
#             {
#                 "stat": stat,
#                 "pvalue": pvalue,
#                 "sim": sim,
#                 "p_equalize": p_equalize,
#                 "p1": p1,
#                 "p2": p2,
#                 "method": "Fisher's",
#             }
#         )
#         stat, pvalue, misc = erdos_renyi_test_paired(A1, A2)
#         rows.append(
#             {
#                 "stat": stat,
#                 "pvalue": pvalue,
#                 "sim": sim,
#                 "p_equalize": p_equalize,
#                 "p1": p1,
#                 "p2": p2,
#                 "method": "McNemar's",
#             }
#         )
# results = pd.DataFrame(rows)


# #%%
# results["detected"] = 0
# results.at[results[results["pvalue"] < 0.05].index, "detected"] = 1

# #%%
# squashed_results = results.groupby(["p1", "p2", "method"]).mean().reset_index()
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# sns.lineplot(data=squashed_results, y="detected", x="p2", hue="method", ax=ax)
# ax.set(ylabel=r"Power (@ $\alpha$ = 0.05)", xlabel="Effect size")
# ax.get_legend().set_title("Test")

# #%%
# fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# from giskard.plot import subuniformity_plot

# subuniformity_plot(
#     results[(results["method"] == "Fisher's") & (results["p1"] == results["p2"])][
#         "pvalue"
#     ],
#     ax=axs[0],
# )

# subuniformity_plot(
#     results[(results["method"] == "McNemar's") & (results["p1"] == results["p2"])][
#         "pvalue"
#     ],
#     ax=axs[1],
# )

# # %%
# from scipy.stats import binom
# from statsmodels.stats.contingency_tables import mcnemar
# from pkg.stats import binom_2samp_paired

# p = 0.1
# n = 1000
# n_same = 10

# pvalues = []
# for i in range(n_sims):
#     samples1 = binom(1, p).rvs(size=100)
#     samples2 = binom(1, p).rvs(size=100)
#     samples2[:n_same] = samples1[:n_same]
#     stat, pvalue, misc = binom_2samp_paired(samples1, samples2)
#     pvalues.append(pvalue)
# pvalues = np.array(pvalues)

# subuniformity_plot(pvalues)
