#%% [markdown]
# # A group-based test

#%%

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from myst_nb import glue as default_glue
from pkg.data import load_matched, load_network_palette
from pkg.io import savefig
from pkg.perturb import remove_edges
from pkg.plot import set_theme
from pkg.stats import stochastic_block_test_paired
from seaborn.utils import relative_luminance

DISPLAY_FIGS = True
FILENAME = "sbm_matched_test"

rng = np.random.default_rng(8888)


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

left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")

#%% [markdown]
# ## A test based on group connection probabilities
#%%
GROUP_KEY = "celltype_discrete"
#%% [markdown]
# ### Run the test
#%%
# TODO double check that all simple groups are the same
stat, pvalue, misc = stochastic_block_test_paired(
    left_adj, right_adj, labels=left_nodes[GROUP_KEY]
)
glue("uncorrected_pvalue", pvalue)

print(pvalue)

#%%
from pkg.plot import plot_pvalues

fig, ax = plot_pvalues(misc, annot_missing=False)

gluefig("pvalue_heatmap_paired_matched", fig)

#%%
from pkg.stats import stochastic_block_test

stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_nodes[GROUP_KEY],
    labels2=right_nodes[GROUP_KEY],
)
glue("uncorrected_pvalue", pvalue)

print(pvalue)


#%%
from pkg.plot import plot_pvalues

plot_pvalues(misc, annot_missing=False)

gluefig("pvalue_heatmap_unpaired_matched", fig)


#%% [markdown]
# ```{glue:figure} fig:sbm_matched_test-uncorrected_pvalues
# :name: "fig:sbm_matched_test-uncorrected_pvalues"
#
# The p-values for each hypothesis test between individual elements of
# the block probability matrices. Each comparison was done by McNemar's test, which
# treats each potential edge as a potential, and examines whether the number of
# disagreeing edges between the left and right is likely to be observed by chance.
# "X" denotes a significant p-value after Bonferroni-Holm correction,
# with $\alpha=0.05$. These individual
# p-values were combined using Fisher's method, resulting in an overall p-value of
# {glue:text}`sbm_matched_test-uncorrected_pvalue:0.2e`.
# ```

#%% [markdown]
# ## Correcting for a global difference in density
#%% [markdown]
# ### Compute the density correction
#%%
n_edges_left = np.count_nonzero(left_adj)
n_edges_right = np.count_nonzero(right_adj)
n_left = left_adj.shape[0]
n_right = right_adj.shape[0]
density_left = n_edges_left / (n_left**2)
density_right = n_edges_right / (n_right**2)

n_remove = int((density_right - density_left) * (n_right**2))

glue("density_left", density_left)
glue("density_right", density_right)
glue("n_remove", n_remove)

#%% [markdown]
# ### Subsample edges on the right hemisphere to set the densities the same
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

gluefig("pvalues_corrected", fig)


#%% [markdown]
# ```{glue:figure} fig:sbm_matched_test-pvalues_corrected
# :name: "fig:sbm_matched_test-pvalues_corrected"

# Histogram of p-values after a correction for network density. For the observed networks
# the left hemisphere has a density of {glue:text}`sbm_matched_test-density_left:0.4f`,
# and the right hemisphere has
# a density of {glue:text}`sbm_matched_test-density_right:0.4f`. Here, we randomly
# removed exactly {glue:text}`n_remove` edges from the right hemisphere network, which
# makes the density of the right network match that of the left hemisphere network.
# Then, we re-ran the stochastic block model testing
# procedure from {numref}`Figure {number} <fig:sbm_matched_test-uncorrected_pvalues>`.
# This entire process
# was repeated {glue:text}`n_resamples` times. The histogram above shows the distribution
# of p-values for the overall test. Note that the p-values are no longer small, indicating
# that with this density correction, we now failed to reject our null hypothesis of
# bilateral symmetry (after density correction) under the stochastic block model.
# ```


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
