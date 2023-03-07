#%%
import numpy as np
from numba import njit
from pkg.data import load_maggot_graph
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP

from graspologic.partition import leiden
from graspologic.plot import networkplot
from graspologic.types import AdjacencyMatrix, Tuple
from graspologic.utils import (
    is_almost_symmetric,
    largest_connected_component,
    pass_to_ranks,
)

mg = load_maggot_graph()

#%%
adj = mg.sum.adj
nodes = mg.nodes

adj, inds = largest_connected_component(adj, return_inds=True)

nodes = nodes.iloc[inds].copy()

adj = pass_to_ranks(adj)

adj = (adj + adj.T) / 2

#%%


@njit
def _project_edges_numba(
    sources: np.ndarray, targets: np.ndarray, weights: np.ndarray, W: np.ndarray
) -> np.ndarray:
    n = W.shape[0]
    k = W.shape[1]
    Z = np.zeros((n, k))
    # TODO redo with broadcasting/einsum?
    for source, target, weight in zip(sources, targets, weights):
        Z[source] += W[target] * weight
        Z[target] += W[source] * weight
    return Z


def _get_edges(adjacency: AdjacencyMatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources, targets = np.nonzero(adjacency)

    # handle the undireced case
    # if undirected, we only need to iterate over the upper triangle of adjacency
    if is_almost_symmetric(adjacency):
        mask = sources <= targets  # includes the diagonal
        sources = sources[mask]
        targets = targets[mask]

    weights = adjacency[sources, targets]

    return sources, targets, weights


def _scale_weights(
    adjacency: AdjacencyMatrix,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    # TODO implement regularized laplacian
    degrees_out = np.sum(adjacency, axis=1)
    degrees_in = np.sum(adjacency, axis=0)

    # regularized laplacian
    degrees_out += degrees_out.mean()
    degrees_in += degrees_in.mean()

    # # may have some cases where these are 0, so set to 1 avoid dividing by 0
    # # doesn't actually mater since these never get multiplied
    # degrees_out[degrees_out == 0] = 1
    # degrees_in[degrees_in == 0] = 1
    degrees_out_root = 1 / np.sqrt(degrees_out)
    degrees_in_root = 1 / np.sqrt(degrees_in)

    weights *= degrees_out_root[sources] * degrees_in_root[targets]
    return weights


def _initialize_projection(features: np.ndarray) -> np.ndarray:
    features_colsum = np.sum(features, axis=0)
    W = features / features_colsum[None, :]
    return W


class GraphEncoderEmbed(BaseEstimator):
    def __init__(self, laplacian: bool = False) -> None:
        """
        Implements the Graph Encoder Embedding of [1]_, which transforms an input
        network and a matrix of node features into a low-dimensional embedding space.
        Parameters
        ----------
        laplacian : bool, optional
            Whether to normalize the embedding by the degree of the input and output
            nodes, by default False
        References
        ----------
        .. [1] C. Shen, Q. Wang, and C. Priebe, "One-Hot Graph Encoder Embedding,"
            arXiv:2109.13098 (2021).
        """
        self.laplacian = laplacian
        super().__init__()

    def fit(
        self, adjacency: AdjacencyMatrix, features: np.ndarray
    ) -> "GraphEncoderEmbed":
        """Fit the embedding model to the input data.
        Parameters
        ----------
        adjacency : AdjacencyMatrix
            n x n adjacency matrix of the graph
        features : np.ndarray
            n x k matrix of node features. These may be one-hot encoded community labels
            or other node features.
        Returns
        -------
        GraphEncoderEmbedding
            The fitted embedding model
        """

        sources, targets, weights = _get_edges(adjacency)

        if self.laplacian:
            weights = _scale_weights(adjacency, sources, targets, weights)

        W = _initialize_projection(features)

        Z = _project_edges_numba(sources, targets, weights, W)

        self.embedding_ = Z
        self.projection_ = W

        return self

    def fit_transform(
        self, adjacency: AdjacencyMatrix, features: np.ndarray
    ) -> np.ndarray:
        """Fit the model to the input data and return the embedding.
        Parameters
        ----------
        adjacency : AdjacencyMatrix
            n x n adjacency matrix of the graph
        features : np.ndarray
            n x k matrix of node features
        Returns
        -------
        np.ndarray
            The n x k embedding of the input graph
        """
        self.fit(adjacency, features)
        return self.embedding_

    def transform(self, adjacency: AdjacencyMatrix) -> np.ndarray:
        """Transform the input adjacency matrix into the embedding space.
        Parameters
        ----------
        adjacency : AdjacencyMatrix
            n x n adjacency matrix of the graph
        Returns
        -------
        np.ndarray
            The n x k embedding of the input graph
        """
        sources, targets, weights = _get_edges(adjacency)

        if self.laplacian:
            weights = _scale_weights(adjacency, sources, targets, weights)

        Z = _project_edges_numba(sources, targets, weights, self.projection_)

        return Z


# %%

leiden_map = leiden(adj, trials=1)

#%%
labels = np.array(list(map(leiden_map.__getitem__, np.arange(adj.shape[0]))))

# %%

gee = GraphEncoderEmbed(laplacian=True)

onehot_labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))

gee_X = gee.fit_transform(adj, onehot_labels)

#%%
from graspologic.embed import LaplacianSpectralEmbed

lse_X = LaplacianSpectralEmbed(n_components=24).fit_transform(adj)

#%%
from graspologic.embed import node2vec_embed

import networkx as nx

g = nx.from_numpy_array(adj)

n2v_out = node2vec_embed(g, dimensions=64, walk_length=30, num_walks=16)

#%%
n2v_X = n2v_out[0]
#%%

umap = UMAP(min_dist=0.9, n_neighbors=15, metric="cosine", random_state=8888)
umap_X = umap.fit_transform(n2v_X)

nodes["x"] = umap_X[:, 0]
nodes["y"] = umap_X[:, 1]

#%%
from graspologic.layouts.auto import _node_positions_from

positions = _node_positions_from(g, np.arange(len(adj)), umap_X)

new_X = []
for pos in positions:
    new_X.append([pos.x, pos.y])
#%%
new_X = np.array(new_X)

nodes["x"] = new_X[:, 0]
nodes["y"] = new_X[:, 1]

#%%
nodes["degree"] = np.sum(adj, axis=1)

#%%
color_key = "celltype_discrete"
#%%
from graspologic.layouts.colors import _get_colors

colors = _get_colors(True, None)["nominal"]

palette = dict(zip(nodes[color_key].unique(), colors))

#%%
import matplotlib.pyplot as plt

ax = networkplot(
    adj,
    node_data=nodes,
    x="x",
    y="y",
    node_hue=color_key,
    node_size="degree",
    palette=palette,
    node_sizes=(10, 80),
    edge_linewidth=0.15,
    edge_alpha=0.05,
)
ax.axis("off")
ax.set_facecolor("white")
fig = ax.get_figure()
fig.set_facecolor("white")
plt.savefig("maggot_networkplot.png", dpi=300, bbox_inches="tight")

# %%
