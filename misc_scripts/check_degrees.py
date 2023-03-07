#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import merge_axes, rotate_labels, soft_axis_off
from graspologic.simulations import sbm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, get_environment_variables
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import (
    SmartSVG,
    draw_hypothesis_box,
    heatmap_grouped,
    make_sequential_colormap,
    networkplot_simple,
    plot_pvalues,
    plot_stochastic_block_probabilities,
    rainbowarrow,
    set_theme,
)
from pkg.stats import stochastic_block_test
from pkg.utils import get_toy_palette, sample_toy_networks
from seaborn.utils import relative_luminance
from svgutils.compose import Figure, Panel, Text


_, _, DISPLAY_FIGS = get_environment_variables()

FILENAME = "check_degrees"

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()
neutral_color = sns.color_palette("Set2")[2]

GROUP_KEY = "celltype_discrete"

left_adj, left_nodes = load_unmatched(side="left")
right_adj, right_nodes = load_unmatched(side="right")

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values

# %%
stat, pvalue, misc = stochastic_block_test(
    left_adj, right_adj, left_labels, right_labels
)

#%%
B1 = misc["probabilities1"]
B2 = misc["probabilities2"]
n1 = misc["group_counts1"]
n2 = misc["group_counts2"]

# %%
from graspologic.simulations import sbm

assert B1.index.equals(n1.index)


# %%


def compute_degrees_df(A, label="Null", sample=None):
    out_degree = np.count_nonzero(A, axis=1)
    out_degree = pd.Series(out_degree, name="out_degree")
    in_degree = np.count_nonzero(A, axis=0)
    in_degree = pd.Series(in_degree, name="in_degree")
    degrees = pd.concat([out_degree, in_degree], axis=1)
    degrees["sample"] = sample
    degrees["label"] = label
    return degrees


dfs = []
n_samples = 100
for i in range(n_samples):
    A = sbm(n1.values, B1.values, directed=True, loops=False)
    degree_df = compute_degrees_df(A, label="Null", sample=i)
    dfs.append(degree_df)

degree_df = compute_degrees_df(left_adj, label="Observed", sample="observed")
dfs.append(degree_df)

degree_df = pd.concat(dfs, axis=0, ignore_index=True)

#%%

colors = sns.color_palette("deep", n_colors=2)
palette = dict(zip(range(n_samples), n_samples * [colors[0]]))
palette["observed"] = colors[1]
palette["Null"] = colors[0]
palette["Observed"] = colors[1]

#%%
set_theme(tick_size=3)
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax = axs[0]
key = "label"
sns.histplot(
    data=degree_df,
    x="out_degree",
    hue=key,
    ax=ax,
    palette=palette,
    stat="density",
    bins=30,
    common_norm=False,
    legend="brief",
    multiple="layer",
    element="bars",
)
sns.move_legend(ax, "upper right", title="Network type", frameon=True)
ax.set_xlabel("Out degree")

ax = axs[1]
sns.histplot(
    data=degree_df,
    x="in_degree",
    hue=key,
    ax=ax,
    palette=palette,
    stat="density",
    bins=30,
    common_norm=False,
    legend="brief",
    multiple="layer",
    element="bars",
)
ax.get_legend().remove()
# sns.move_legend(ax, "upper right", title='Network type')
ax.set_xlabel("In degree")

gluefig("degree_hists", fig)

# %%

from graspologic.models import DCEREstimator, EREstimator

dcer = DCEREstimator(directed=True, loops=False, degree_directed=True)
er = EREstimator(directed=True, loops=False)

dcer.fit(left_adj)
er.fit(left_adj)

#%%
n_samples = 1000

from pkg.stats import compute_density
from tqdm.autonotebook import tqdm

rows = []
for model, model_name in zip([dcer, er], ["DCER", "ER"]):
    samples = model.sample(n_samples)
    for sample in tqdm(samples):
        density = compute_density(sample)
        rows.append({"model": model_name, "sample": sample, "density": density})

results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.histplot(data=results, x="density", hue="model", ax=ax)

sns.move_legend(ax, "upper right", title="Model", frameon=True)

gluefig("density_hists_by_model", fig)

# %%

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
key = "sample"
# sns.histplot(
#     data=degree_df,
#     x="out_degree",
#     hue=key,
#     ax=ax,
#     palette=palette,
#     stat="density",
#     bins=30,
#     common_norm=False,
# )
sns.kdeplot(
    data=degree_df.query('label == "Null"'),
    x="out_degree",
    hue=key,
    ax=ax,
    palette=palette,
    common_norm=False,
    clip=(0, 1000),
    lw=0.1,
)
sns.kdeplot(
    data=degree_df.query('label == "Observed"'),
    x="out_degree",
    hue=key,
    ax=ax,
    palette=palette,
    clip=(0, 1000),
    lw=3,
)
ax.get_legend().remove()
ax.set_xlabel("Out degree")
gluefig("out_degree_kdes", fig)

# %%
