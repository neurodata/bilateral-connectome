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


from scipy.stats import binom
from pkg.stats import binom_2samp
from tqdm.autonotebook import tqdm

_, RERUN_SIMS, DISPLAY_FIGS = get_environment_variables()

FILENAME = "single_subgraph_power"

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

n_nodes_range = np.linspace(5, 500, 40, dtype=int)
print(n_nodes_range)
# base_p_range = [0.005, 0.01, 0.05, 0.1, 0.5]
base_p_range = np.geomspace(0.005, 0.5, 40)
print(base_p_range)
n_sims = 1000

effect_scale = 0.8
method = "fisher"

#%%

if RERUN_SIMS:
    pbar = tqdm(total=len(n_nodes_range) * len(base_p_range) * n_sims)
    rows = []
    for n in n_nodes_range:
        for base_p in base_p_range:
            for sim in range(n_sims):

                # would be n*(n-1) if we didn't count self-edges for an induced subgraph
                # trying to keep things general here
                possible_edges = n**2
                edges1 = binom.rvs(possible_edges, base_p, random_state=rng)
                edges2 = binom.rvs(
                    possible_edges, effect_scale * base_p, random_state=rng
                )
                stat, pvalue = binom_2samp(
                    edges1, possible_edges, edges2, possible_edges, method=method
                )
                rows.append(
                    {
                        "n": n,
                        "base_p": base_p,
                        "sim": sim,
                        "stat": stat,
                        "pvalue": pvalue,
                    }
                )

                pbar.update(1)

    pbar.close()
    results = pd.DataFrame(rows)
    results.to_csv(
        "bilateral-connectome/results/outputs/single_subgraph_power/sim_results.csv"
    )
else:
    results = pd.read_csv(
        "bilateral-connectome/results/outputs/single_subgraph_power/sim_results.csv",
        index_col=0,
    )


# %%


def compute_power(pvalues, alpha=0.05):
    return np.mean(pvalues < alpha)


power_results = (
    results.groupby(["n", "base_p"])["pvalue"]
    .apply(compute_power)
    .rename("power")
    .reset_index()
)

square_power = power_results.pivot(index="base_p", columns="n", values="power")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(square_power, ax=ax, cmap="RdBu_r", center=0, vmin=0, vmax=1)
ax.invert_yaxis()

#%%
from scipy.interpolate import interp1d, RegularGridInterpolator

cols = square_power.columns.values
rows = square_power.index.values
values = square_power.values
interpolator = RegularGridInterpolator((rows, cols), values, bounds_error=True)

new_row_range = np.geomspace(0.005, 0.5, 400)
new_col_range = np.linspace(5, 500, 400)

rows, cols = np.meshgrid(new_row_range, new_col_range, indexing="ij")
zs = interpolator((rows, cols))

interp_df = pd.DataFrame(data=zs, index=new_row_range, columns=new_col_range)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(interp_df, cmap="RdBu_r", center=0, vmin=0, vmax=1)

#%%

set_theme(tick_size=5)

df = interp_df

levels = [0.25, 0.5, 0.9]
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    df,
    cmap="RdBu_r",
    center=0,
    vmin=0,
    vmax=1,
    square=True,
    cbar_kws=dict(shrink=0.6, pad=0.1),
)
ax.invert_yaxis()
cs = plt.contour(zs, levels=levels, colors="black")
ax.clabel(cs, cs.levels, manual=[(0, 100), (100, 90), (463, 370)], colors="black")


iloc_to_x_value = interp1d(
    np.arange(0, df.shape[1]) + 0.5,
    df.columns.values,
    kind="linear",
    bounds_error=False,
    fill_value=(df.columns.values.min(), df.columns.values.max()),
)
x_value_to_iloc = interp1d(
    df.columns.values,
    np.arange(0, df.shape[1]) + 0.5,
    kind="linear",
    bounds_error=False,
    fill_value=(0.5, df.shape[1] - 0.5),
)

iloc_to_y_value = interp1d(
    np.arange(0, df.shape[0]) + 0.5,
    df.index.values,
    kind="linear",
    bounds_error=False,
    fill_value=(df.index.values.min(), df.index.values.max()),
)
y_value_to_iloc = interp1d(
    df.index.values,
    np.arange(0, df.shape[0]) + 0.5,
    kind="linear",
    bounds_error=False,
    fill_value=(-10, df.shape[0] + 10),  # not sure what these should be
)
ax.set(xticks=[], yticks=[], xlabel="", ylabel="")

xax2 = ax.secondary_xaxis(0, functions=(iloc_to_x_value, x_value_to_iloc))
xax2.set_xticks([5, 125, 250, 375, 500])


from matplotlib.ticker import ScalarFormatter

# TODO not sure why the boundaries are not showing up exactly correctly.
yax2 = ax.secondary_yaxis(0, functions=(iloc_to_y_value, y_value_to_iloc))
yax2.set_yscale("log")
yax2.set_yticks([0.005, 0.01, 0.05, 0.1, 0.5])
yax2.get_yaxis().set_major_formatter(ScalarFormatter())
yax2.set_ylim(0.005, 0.5)

cax = fig.get_axes()[1]
cax.set_title("Power @\n" + r"$\alpha=0.05$", pad=20)

yax2.set_ylabel("Base connection probability")
xax2.set_xlabel("Number of nodes")

gluefig("power_heatmap_contours", fig)


#%%


def contour(X, Y, text, offset=0):

    # Interpolate text along curve
    # X0,Y0 for position  + X1,Y1 for normal vectors
    path = TextPath(
        (0, -0.75), text, prop=FontProperties(size=2, family="Roboto", weight="bold")
    )
    V = path.vertices
    X0, Y0, D = interpolate(X, Y, offset + V[:, 0])
    X1, Y1, _ = interpolate(X, Y, offset + V[:, 0] + 0.1)

    # Here we interpolate the original path to get the "remainder"
    #  (path minus text)
    X, Y, _ = interpolate(X, Y, np.linspace(V[:, 0].max() + 1, D - 1, 200))
    plt.plot(
        X, Y, color="black", linewidth=0.5, markersize=1, marker="o", markevery=[0, -1]
    )

    # Transform text vertices
    dX, dY = X1 - X0, Y1 - Y0
    norm = np.sqrt(dX**2 + dY**2)
    dX, dY = dX / norm, dY / norm
    X0 += -V[:, 1] * dY
    Y0 += +V[:, 1] * dX
    V[:, 0], V[:, 1] = X0, Y0

    # Faint outline
    patch = PathPatch(
        path,
        facecolor="white",
        zorder=10,
        alpha=0.25,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.add_artist(patch)

    # Actual text
    patch = PathPatch(
        path, facecolor="black", zorder=30, edgecolor="black", linewidth=0.0
    )
    ax.add_artist(patch)


for level, collection in zip(cs.levels[:], cs.collections[:]):
    for path in collection.get_paths():
        V = np.array(path.vertices)
        text = "%.3f" % level
        contour(V[:, 0], V[:, 1], text)


#%%
levels = 8
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(square_power, cmap="RdBu_r", center=0, vmin=0, vmax=1)
plt.contour(square_power, levels=levels)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.contourf(zs, levels=levels)
plt.contour(zs, levels=levels)

#%%
from scipy.ndimage import gaussian_filter

filtered = gaussian_filter(square_power, sigma=(0.002, 2))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(filtered, cmap="RdBu_r", center=0, vmin=0, vmax=1)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.contourf(filtered, levels=levels)
plt.contour(filtered, levels=levels)

#%%

set_theme(tick_size=5)

df = square_power


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(df, cmap="RdBu_r", center=0, vmin=0, vmax=1, ax=ax)
ax.invert_yaxis()

iloc_to_x_value = interp1d(
    np.arange(0, df.shape[1]) + 0.5,
    df.columns.values,
    kind="linear",
    bounds_error=False,
    fill_value=(df.columns.values.min(), df.columns.values.max()),
)
x_value_to_iloc = interp1d(
    df.columns.values,
    np.arange(0, df.shape[1]) + 0.5,
    kind="linear",
    bounds_error=False,
    fill_value=(0.5, df.shape[1] - 0.5),
)

iloc_to_y_value = interp1d(
    np.arange(0, df.shape[0]) + 0.5,
    df.index.values,
    kind="linear",
    bounds_error=False,
    fill_value=(df.index.values.min(), df.index.values.max()),
)
y_value_to_iloc = interp1d(
    df.index.values,
    np.arange(0, df.shape[0]) + 0.5,
    kind="linear",
    bounds_error=False,
    fill_value=(-10, df.shape[0] + 10),  # not sure what these should be
)
ax.set(xticks=[], yticks=[], xlabel="", ylabel="")

xax2 = ax.secondary_xaxis(0, functions=(iloc_to_x_value, x_value_to_iloc))
xax2.set_xticks([5, 250, 500])


from matplotlib.ticker import ScalarFormatter

# TODO not sure why the boundaries are not showing up exactly correctly.
yax2 = ax.secondary_yaxis(0, functions=(iloc_to_y_value, y_value_to_iloc))
yax2.set_yscale("log")
yax2.set_yticks([0.005, 0.01, 0.05, 0.1, 0.5])
yax2.get_yaxis().set_major_formatter(ScalarFormatter())
yax2.set_ylim(0.005, 0.5)

# xax2.set_xticklabels(["5", "250", "500"])

# basically fitting splines to interpolate linearly between points we checked
# prop_to_thresh = interp1d(
#     x=x, y=y, kind="slinear", bounds_error=False, fill_value=(0, 1)
# )
# thresh_to_prop = interp1d(
#     x=y, y=x, kind="slinear", bounds_error=False, fill_value=(0, 1)
# )

# ax2 = ax.secondary_xaxis(-0.2, functions=(prop_to_thresh, thresh_to_prop))

# if weight == "input_proportion":
#     ax2.set_xticks([0.005, 0.01, 0.015, 0.02])
#     ax2.set_xticklabels(["0.5%", "1%", "1.5%", "2%"])
#     ax2.set_xlabel("Weight threshold (input percentage)")
# elif weight == "synapse_count":
#     ax2.set_xlabel("Weight threshold (synapse count)")
# ax2.tick_params(axis="both", length=5)

# %%
