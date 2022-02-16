import numpy as np


def compute_density(adjacency, loops=False):
    n_edges = np.count_nonzero(adjacency)
    n_nodes = adjacency.shape[0]
    n_possible = n_nodes ** 2
    if not loops:
        n_possible -= n_nodes
    return n_edges / n_possible


def compute_density_adjustment(adjacency1, adjacency2):
    density1 = compute_density(adjacency1)
    density2 = compute_density(adjacency2)
    return density1 / density2
