#%% [markdown]
# # A group-based test

#%%

from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from myst_nb import glue as default_glue
from pkg.data import load_matched, load_network_palette, load_node_palette
from pkg.io import savefig
from pkg.perturb import remove_edges
from pkg.plot import set_theme
from pkg.stats import stochastic_block_test_paired

DISPLAY_FIGS = True
FILENAME = "sbm_matched_test"

rng = np.random.default_rng(8888)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue("fig:" + name, fig)

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var):
    default_glue(f"{FILENAME}-{name}", var, display=False)


t0 = time.time()
set_theme(font_scale=1.25)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()

left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")


#%%

# TODO double check that all simple groups are the same
stat, pvalue, misc = stochastic_block_test_paired(
    left_adj, right_adj, labels=left_nodes["simple_group"]
)
glue("pvalue", pvalue)

n_no_edge = misc["neither"]
n_both_edge = misc["both"]
n_only_left = misc["only1"]
n_only_right = misc["only2"]


#%%
n_edges_left = np.count_nonzero(left_adj)
n_edges_right = np.count_nonzero(right_adj)
n_left = left_adj.shape[0]
n_right = right_adj.shape[0]
density_left = n_edges_left / (n_left ** 2)
density_right = n_edges_right / (n_right ** 2)

n_remove = int((density_right - density_left) * (n_right ** 2))

glue("density_left", density_left)
glue("density_right", density_right)
glue("n_remove", n_remove)


#%%

rows = []
n_resamples = 25
glue("n_resamples", n_resamples)
for i in range(n_resamples):
    subsampled_right_adj = remove_edges(
        right_adj, effect_size=n_remove, random_seed=rng
    )
    stat, pvalue, misc = stochastic_block_test_paired(
        left_adj, subsampled_right_adj, labels=left_nodes["simple_group"]
    )
    rows.append({"stat": stat, "pvalue": pvalue, "misc": misc, "resample": i})

resample_results = pd.DataFrame(rows)


#%% [markdown]
# ### Plot the p-values for the corrected tests
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(data=resample_results, x="pvalue", ax=ax)
ax.set(xlabel="p-value", ylabel="", yticks=[])
ax.spines["left"].set_visible(False)

mean_resample_pvalue = np.mean(resample_results["pvalue"])
median_resample_pvalue = np.median(resample_results["pvalue"])

gluefig("pvalues-corrected", fig)

#%%
# %%
# ```{glue:figure} fig:pvalues-corrected
# :name: "fig:pvalues-corrected"

# Histogram of p-values after a correction for network density. For the observed networks
# the left hemisphere has a density of {glue:text}`density_left:0.4f`, and the right hemisphere has
# a density of {glue:text}`density_right:0.4f`. Here, we randomly removed exactly {glue:text}`n_remove`
# edges from the right hemisphere network, which makes the density of the right network
# match that of the left hemisphere network. Then, we re-ran the stochastic block model testing
# procedure from {numref}`Figure {number} <fig:sbm-uncorrected>`. This entire process
# was repeated {glue:text}`n_resamples` times. The histogram above shows the distribution
# of p-values for the overall test. Note that the p-values are no longer small, indicating
# that with this density correction, we now failed to reject our null hypothesis of
# bilateral symmetry under the stochastic block model.
# ```


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
