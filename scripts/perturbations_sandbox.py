#%%

from networkx.classes.function import non_edges
import numpy as np
from sklearn.utils import check_random_state
from graspologic.simulations import er_np
from graspologic.plot import heatmap
from numpy.random import default_rng
from numpy.random import SeedSequence
from numba import jit


n = 20
p = 0.1

A = er_np(n, p, directed=True, loops=False)
heatmap(A)

#%%


def _input_checks(adjacency, random_seed, effect_size, max_tries):
    adjacency = adjacency.copy()
    n = adjacency.shape[0]
    rng = default_rng(random_seed)

    if max_tries is None:
        max_tries = effect_size * 10

    return adjacency, n, rng, max_tries


def remove_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
    adjacency, n, rng, max_tries = _input_checks(
        adjacency, random_seed, effect_size, max_tries
    )

    nonzero_inds = np.nonzero(adjacency)
    edge_counter = np.arange(len(nonzero_inds[0]))
    rng.shuffle(edge_counter)
    for i in edge_counter[:effect_size]:
        source = nonzero_inds[0][i]
        target = nonzero_inds[1][i]
        adjacency[source, target] = 0
    return adjacency


# @jit(nopython=True)
def add_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
    # TODO https://numba-how-to.readthedocs.io/en/latest/numpy.html
    adjacency, n, rng, max_tries = _input_checks(
        adjacency, random_seed, effect_size, max_tries
    )

    n_edges_added = 0
    tries = 0
    while n_edges_added < effect_size and tries < max_tries:
        i, j = rng.integers(0, n, size=2)
        tries += 1
        if i != j and adjacency[i, j] == 0:
            adjacency[i, j] = 1
            n_edges_added += 1

    if tries == max_tries:
        msg = (
            "Maximum number of tries reached when adding edges, number added was"
            " less than specified."
        )
        raise UserWarning(msg)

    return adjacency


B = add_edges(A, effect_size=1)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

heatmap(A, ax=axs[0])
heatmap(B, ax=axs[1])
heatmap(A - B, ax=axs[2])

#%%

from giskard.utils import get_random_seed


def shuffle_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
    rng = default_rng(random_seed)

    seed = get_random_seed(rng)
    adjacency = remove_edges(
        adjacency, effect_size=effect_size, random_seed=seed, max_tries=max_tries
    )

    seed = get_random_seed(rng)
    adjacency = add_edges(
        adjacency, effect_size=effect_size, random_seed=seed, max_tries=max_tries
    )

    return adjacency


B = shuffle_edges(A, effect_size=5)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

heatmap(A, ax=axs[0])
heatmap(B, ax=axs[1])
heatmap(A - B, ax=axs[2])

#%%

