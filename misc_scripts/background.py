#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pkg.plot import set_theme
from graspologic.simulations import sample_edges
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel

from pkg.io import glue as default_glue
from pkg.io import savefig

set_theme()

DISPLAY_FIGS = False

FILENAME = "background"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


#%%

rng = np.random.default_rng(888888)

width = 16
height = 9
n_nodes = 100

ys = rng.uniform(0, height, size=n_nodes)
xs = rng.uniform(0, width, size=n_nodes)

fig, ax = plt.subplots(1, 1, figsize=(width, height))


sns.scatterplot(x=xs, y=ys, ax=ax)

#%%

X = np.stack((xs, ys), axis=1)
dists = rbf_kernel(X, gamma=1)

sns.heatmap(dists)

#%%

P = dists * 8
P[P >= 1] = 1
np.random.seed(88)
A = sample_edges(P)

#%%
g = nx.from_numpy_array(A)

pos = dict(zip(np.arange(n_nodes), list(zip(xs, ys))))
fig, ax = plt.subplots(1, 1, figsize=(width, height))

nx.draw_networkx(g, pos=pos, ax=ax, with_labels=False)

# %%

from graspologic.plot import networkplot

fig, ax = plt.subplots(1, 1, figsize=(width, height))

networkplot(
    A,
    x=xs,
    y=ys,
    ax=ax,
    edge_linewidth=1.5,
    edge_alpha=0.7,
    node_alpha=1.0,
    node_kws=dict(linewidth=1, edgecolor="black"),
)

ax.axis("off")
ax.set(xlim=(xs.min(), xs.max()), ylim=(ys.min(), ys.max()))

# sns.scatterplot(x=xs, y=ys, ax=ax)

gluefig("background", fig)
