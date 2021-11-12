#%% [markdown]
# # A density-based test
# Here, we compare the two unmatched networks by treating each as an Erdos-Renyi network
# and simply comparing their estimated densities.

#%% [markdown]
# ## The Erdos-Renyi model
# The Erdos-Renyi (ER) model is one of the simplest network models. This model treats
# the probability of each potential edge in the network occuring to be the same. In
# other words, all edges between any two nodes are equally likely.
#
# ```{admonition} Math
# We say that for all $(i, j), i \neq j$, with $i$ and $j$ both running
# from $1 ... n$, the probability of the edge $(i, j)$ occuring is:
#
# $$ P[A_{ij} = 1] = P_{ij} = p $$
#
# ```
#
# Thus, for this model, the only parameter of interest is the global connection
# probability, $p$. This is sometimes also referred to as the *network density*.
#
# In order to compare two networks $A_{left}$ and $A_{right}$ under this model, we
# simply need to compute these network densities ($p_{left}$ and $p_{right}$), and then
# run a statistical test to see if these densities are significantly different.
#
# ```{admonition} Math
# Under this
# model, the total number of edges $m$ comes from a $Binomial(n^2 - n, p)$ distribution,
# where $n$ is the number of nodes. If $m_{left}$ is the number of edges on the left
# hemisphere, and $m_{right}$ is the number of edges on the right, then we have:
#
# $$m_{left} \sim Binomial(n_{left}^2 - n_{left}, p_{left})$$
#
# and independently,
#
# $$m_{right} \sim Binomial(n_{right}^2 - n_{right}, p_{right})$$
#
# To compare the two networks, we are just interested in a comparison of $p_{left}$ vs.
# $p_{right}$. Formally, we are testing:
#
# $$H_0: p_{left} = p_{right}, \quad H_a: p_{left} \neq p_{right}$$
#
# Fortunately, the problem of testing for equal proportions is well studied.
# In our case, we'll use Fisher's Exact test to run this test for the null and
# alternative hypotheses above.
# ```

#%% [markdown]
# ## Preliminaries
#%%

from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import erdos_renyi_test
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint

DISPLAY_FIGS = False

FILENAME = "er_unmatched_test"


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
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()


left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

#%% [markdown]
# ## A test based on network densities
#%% [markdown]
# ### Compute the density and other parameters
#%%
n_nodes_left = left_adj.shape[0]
n_nodes_right = right_adj.shape[0]
n_possible_left = n_nodes_left ** 2 - n_nodes_left
n_possible_right = n_nodes_right ** 2 - n_nodes_right
n_edges_left = np.count_nonzero(left_adj)
n_edges_right = np.count_nonzero(right_adj)
density_left = n_edges_left / n_possible_left
density_right = n_edges_right / n_possible_right

glue("density_left", density_left)
glue("density_right", density_right)

left_binom = binom(n_possible_left, density_left)
right_binom = binom(n_possible_right, density_right)

#%% [markdown]
# ### Plot the densities
#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))


ax.bar(0, density_left, color=network_palette["Left"])
ax.bar(1, density_right, color=network_palette["Right"])

coverage = 0.99
coverage_percentage = coverage * 100
glue("coverage_percentage", coverage_percentage)
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

gluefig("er_density", fig)

#%% [markdown]
# ### Run the test to compare proportions
#%%
stat, pvalue, _ = erdos_renyi_test(left_adj, right_adj)
glue("pvalue", pvalue)


#%% [markdown]

# ```{glue:figure} fig:er_unmatched_test-er_density
# :name: "fig:er_unmatched_test-er_density"
#
# Comparison of estimated densities for the left and right hemisphere networks. The
# estimated density (probability of any edge across the entire network), $\hat{p}$, for
# the left
# hemisphere is {glue:text}`er_unmatched_test-density_left:0.3f`, while for the right
# it is
# {glue:text}`er_unmatched_test-density_right:0.3f`. Black lines denote
# {glue:text}`er_unmatched_test-coverage_percentage`**%**
# confidence intervals for this estimated parameter $\hat{p}$. The p-value for testing
# the null hypothesis that these densities are the same is
# {glue:text}`er_unmatched_test-pvalue:0.3g` (two
# sided Fisher's exact test).
# ```

#%% [markdown]
# ## End

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
