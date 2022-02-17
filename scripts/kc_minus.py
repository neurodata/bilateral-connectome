#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, savefig
from pkg.plot import plot_pvalues, set_theme
from pkg.stats import erdos_renyi_test, stochastic_block_test

DISPLAY_FIGS = True

FILENAME = "kc_minus"


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


#%%
t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()
neutral_color = sns.color_palette("Set2")[2]

GROUP_KEY = "simple_group"

left_adj, left_nodes = load_unmatched(side="left")
right_adj, right_nodes = load_unmatched(side="right")

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values

left_nodes["inds"] = range(len(left_nodes))
sub_left_nodes = left_nodes[left_nodes[GROUP_KEY] != "KCs"]
sub_left_inds = sub_left_nodes["inds"].values
right_nodes["inds"] = range(len(right_nodes))
sub_right_nodes = right_nodes[right_nodes[GROUP_KEY] != "KCs"]
sub_right_inds = sub_right_nodes["inds"].values

sub_left_adj = left_adj[np.ix_(sub_left_inds, sub_left_inds)]
sub_right_adj = right_adj[np.ix_(sub_right_inds, sub_right_inds)]
sub_left_labels = sub_left_nodes[GROUP_KEY]
sub_right_labels = sub_right_nodes[GROUP_KEY]

#%%

# # TODO fix this
# stat, pvalue, misc = stochastic_block_test(
#     left_adj,
#     right_adj,
#     labels1=left_labels,
#     labels2=right_labels,
#     method="fisher",
#     combine_method="tippett",
# )
# pvalue_vmin = np.log10(np.nanmin(misc["uncorrected_pvalues"].values))


stat, pvalue, misc = erdos_renyi_test(sub_left_adj, sub_right_adj)
print(pvalue)
glue("er_pvalue", pvalue)

stat, pvalue, misc = stochastic_block_test(
    sub_left_adj,
    sub_right_adj,
    labels1=sub_left_labels,
    labels2=sub_right_labels,
    method="fisher",
    combine_method="tippett",
)
print(pvalue)
glue("sbm_pvalue", pvalue)

fig, ax = plot_pvalues(misc)

#%%

stat, pvalue, misc = stochastic_block_test(
    sub_left_adj,
    sub_right_adj,
    labels1=sub_left_labels,
    labels2=sub_right_labels,
    method="fisher",
    density_adjustment=True,
    combine_method="tippett",
)
print(pvalue)
glue("asbm_pvalue", pvalue)
glue("asbm_pvalue_formatted", f"{pvalue:.2g}")
# fig, ax = plot_pvalues(misc, pvalue_vmin)
