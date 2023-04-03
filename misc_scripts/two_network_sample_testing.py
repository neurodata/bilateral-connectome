#%%
from pkg.utils import set_warnings

set_warnings()

import datetime
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.simulations import er_corr, er_np
from myst_nb import glue as default_glue
from pkg.data import load_network_palette
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import set_warnings
from pkg.plot import soft_axis_off

DISPLAY_FIGS = True

FILENAME = "two_network_sample_testing"


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


set_theme(font_scale=1.5)

network_palette, _ = load_network_palette()

#%%
p = 0.3
glue("p", p)

rho = 0.9
glue("rho", rho)

# sample two networks
np.random.seed(888)

A1 = er_np(16, p)
A2 = er_np(10, p)


g1 = nx.from_numpy_array(A1)
g2 = nx.from_numpy_array(A2)


fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw=dict(wspace=0))

for ax in axs:
    soft_axis_off(ax)

draw_kws = dict(with_labels=False, node_size=100)

ax = axs[0]
pos1 = nx.kamada_kawai_layout(g1)
nx.draw_networkx(
    g1,
    pos=pos1,
    ax=ax,
    node_color=network_palette["Left"],
    edge_color=network_palette["Left"],
    **draw_kws,
)
ax.set_xlabel("Left", color=network_palette["Left"])

ax = axs[1]
pos2 = nx.kamada_kawai_layout(g2)
nx.draw_networkx(
    g2,
    pos=pos2,
    ax=ax,
    node_color=network_palette["Right"],
    edge_color=network_palette["Right"],
    **draw_kws,
)
ax.set_xlabel("Right", color=network_palette["Right"])


# ax.axis("off")

fig.set_facecolor("white")  # not sure why necessary in VS code for this plot

gluefig("2_network_sample_testing", fig)

# %%
