#%% [markdown]
# # A density-based test
# Here, we compare the two unmatched networks by treating each as an Erdos-Renyi network
# and simply comparing their estimated densities.

#%%

from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import erdos_renyi_test
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint

DISPLAY_FIGS = False


def gluefig(name, fig, **kwargs):
    foldername = "er_unmatched_test"
    savefig(name, foldername=foldername, **kwargs)

    glue("fig:" + name, fig, display=False)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()


left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

#%%
n_nodes_left = left_adj.shape[0]
n_nodes_right = right_adj.shape[0]
n_possible_left = n_nodes_left ** 2 - n_nodes_left
n_possible_right = n_nodes_right ** 2 - n_nodes_right
n_edges_left = np.count_nonzero(left_adj)
n_edges_right = np.count_nonzero(right_adj)
density_left = n_edges_left / n_possible_left
density_right = n_edges_right / n_possible_right

glue("density_left", density_left, display=False)
glue("density_right", density_right, display=False)

left_binom = binom(n_possible_left, density_left)
right_binom = binom(n_possible_right, density_right)

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))


ax.bar(0, density_left, color=network_palette["Left"])
ax.bar(1, density_right, color=network_palette["Right"])

coverage = 0.99
coverage_percentage = coverage * 100
glue("coverage_percentage", coverage_percentage, display=False)
left_lower, left_upper = proportion_confint(
    n_edges_left, n_possible_left, alpha=1 - coverage, method="beta"
)
right_lower, right_upper = proportion_confint(
    n_edges_right, n_possible_right, alpha=1 - coverage, method="beta"
)

ax.plot([0, 0], [left_lower, left_upper], color="black", linewidth=4)
ax.plot([1, 1], [right_lower, right_upper], color="black", linewidth=4)

ax.set(
    xlabel="Network",
    xticks=[0, 1],
    xticklabels=["Left", "Right"],
    ylabel=r"Estimated density ($\hat{p}$)",
)

gluefig("er-density", fig)

#%%
stat, pvalue, _ = erdos_renyi_test(left_adj, right_adj)
glue("pvalue", pvalue, display=False)


#%% [markdown]

# ```{glue:figure} fig:er-density
# :name: "fig:er-density"
#
# Comparison of estimated densities for the left and right hemisphere networks. The
# estimated density (probability of any edge across the entire network), $\hat{p}$, for
# the left
# hemisphere is {glue:text}`density_left:0.3f`, while for the right it is
# {glue:text}`density_right:0.3f`. Black lines denote {glue:text}`coverage_percentage`**%**
# confidence intervals for this estimated parameter $\hat{p}$. The p-value for testing
# the null hypothesis that these densities are the same is {glue:text}`pvalue:0.3g` (two
# sided Fisher's exact test).
# ```

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
