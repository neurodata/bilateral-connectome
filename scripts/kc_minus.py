#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import rotate_labels
from graspologic.plot import networkplot
from matplotlib.transforms import Bbox
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, savefig
from pkg.perturb import remove_edges
from pkg.plot import bound_points, set_theme
from pkg.stats import erdos_renyi_test, stochastic_block_test
from seaborn.utils import relative_luminance


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

from pkg.stats import compute_density_adjustment

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

# n_edges_left = np.count_nonzero(sub_left_adj)
# n_edges_right = np.count_nonzero(sub_right_adj)
# n_left = sub_left_adj.shape[0]
# n_right = sub_right_adj.shape[0]
# density_left = n_edges_left / (n_left ** 2)
# density_right = n_edges_right / (n_right ** 2)

null_ratio = compute_density_adjustment(sub_left_adj, sub_right_adj)

stat, pvalue, misc = stochastic_block_test(
    sub_left_adj,
    sub_right_adj,
    labels1=sub_left_labels,
    labels2=sub_right_labels,
    method="fisher",
    null_odds=null_ratio,
    combine_method="tippett",
)
print(pvalue)
glue("asbm_pvalue", pvalue)
glue("asbm_pvalue_formatted", f"{pvalue:.2g}")
