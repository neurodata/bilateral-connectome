#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from sklearn.neighbors import kneighbors_graph
from matplotlib.patches import Polygon
from graspologic.plot import networkplot
from pkg.plot import set_theme

from pkg.io import glue as default_glue
from pkg.io import savefig

set_theme()

DISPLAY_FIGS = True

FILENAME = "draw_brain_comparisons"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


#%%

# REF: https://scidraw.io/drawing/417
image_path = "bilateral-connectome/data/images/drosophila-brain-outline.png"

#%%


def find_contour(image_path):
    image_grid = plt.imread(image_path)
    max_project = image_grid.sum(axis=2)
    non_background = max_project != 4
    contours = find_contours(non_background[::-1], level=0.95)
    contour = contours[0]
    contour = np.row_stack((contour, contour[0].reshape(1, -1)))
    contour = np.column_stack((contour[:, 1], contour[:, 0]))
    return contour


def sample_spatial_graph(n_points=200, n_neighbors=5, seed=8888):
    rng = np.random.default_rng(seed)
    locs_x = rng.uniform(0, 3500, size=n_points)
    locs_y = rng.uniform(0, 1800, size=n_points)
    X = np.column_stack((locs_x, locs_y))

    A = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
    )
    return A, X


def draw_network_brain(
    n_points=125,
    n_neighbors=5,
    seed=888888,
    pad=10,
    edge_linewidth=2.5,
    node_linewidth=0.75,
    node_edgecolor="darkgrey",
    node_size=100,
    node_color="#1f78b4",
    edge_color="#a6cee3",
    outline_color="lightgrey",
    outline_edgecolor="#7570b3",
    outline_alpha=0.5,
    outline_linewidth=6,
    ax=None,
    figsize=(10, 5),
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    contour = find_contour(image_path)
    A, X = sample_spatial_graph(n_points=n_points, n_neighbors=n_neighbors, seed=seed)

    networkplot(
        A,
        x=X[:, 0],
        y=X[:, 1],
        ax=ax,
        edge_linewidth=edge_linewidth,
        edge_alpha=1.0,
        node_alpha=1.0,
        node_kws=dict(
            linewidth=node_linewidth,
            edgecolor=node_edgecolor,
            s=node_size,
            color=node_color,
        ),
        edge_kws=dict(color=edge_color),
    )
    nodes = ax.get_children()[0]
    edges = ax.get_children()[1]

    ax.plot(
        contour[:, 0],
        contour[:, 1],
        linewidth=outline_linewidth,
        color=outline_edgecolor,
    )
    ax.set(
        xlim=(contour[:, 0].min() - pad, contour[:, 0].max() + pad),
        ylim=(contour[:, 1].min() + pad, contour[:, 1].max() + pad),
    )
    poly = Polygon(contour, color=outline_color, zorder=-5, alpha=outline_alpha)
    ax.add_patch(poly)

    nodes.set_clip_path(poly)
    edges.set_clip_path(poly)

    ax.axis("off")

    misc = dict(X=X, A=A, contour=contour)

    return fig, ax, misc


fig, ax, _ = draw_network_brain(n_points=125, seed=888888, n_neighbors=5, node_size=100)
gluefig("brain_test", fig)

#%%
fig, axs = plt.subplots(
    1, 3, figsize=(22, 5), gridspec_kw=dict(width_ratios=[1, 0.1, 1])
)
_, _, misc1 = draw_network_brain(ax=axs[0])
ax = axs[1]
ax.text(0.2, 0.4, r"$\overset{?}{\approx}$", fontsize=100, va="center", ha="center")
ax.axis("off")
_, _, misc2 = draw_network_brain(ax=axs[2], seed=888)
gluefig("brain_approx_equals", fig)

#%%
from matplotlib.patches import ConnectionPatch

fig, axs = plt.subplots(
    1, 3, figsize=(22, 5), gridspec_kw=dict(width_ratios=[1, 0.1, 1])
)
_, _, misc1 = draw_network_brain(ax=axs[0])
ax = axs[1]
ax.axis("off")
_, _, misc2 = draw_network_brain(ax=axs[2], seed=888888)
# axs[0].axhline(1000)
n_lines = 10
X1 = misc1["X"]
X2 = misc2["X"]
rng = np.random.default_rng(888)
for line in range(n_lines):
    point1 = (-100, 100000)
    while (point1[0] < 2400) or (point1[1] > 1100):
        point1 = X1[rng.choice(len(X1))]
    point2 = (1000, -100)
    while (point2[0] > 500) or (point2[1] > 1100):
        point2 = X2[rng.choice(len(X2))]
    con = ConnectionPatch(
        xyA=point1,
        xyB=point2,
        coordsA="data",
        coordsB="data",
        axesA=axs[0],
        axesB=axs[2],
        linestyle='--',
        linewidth=3,
        zorder=10
    )
    axs[2].add_artist(con)
gluefig('brain_matching', fig)