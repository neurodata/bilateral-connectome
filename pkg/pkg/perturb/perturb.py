import numpy as np
from numpy.random import default_rng
from giskard.utils import get_random_seed


def _input_checks(adjacency, random_seed, effect_size, max_tries):
    adjacency = adjacency.copy()
    n = adjacency.shape[0]

    if isinstance(random_seed, np.integer):
        rng = default_rng(random_seed)
    else:
        rng = random_seed

    if max_tries is None:
        max_tries = effect_size * 10

    return adjacency, n, rng, max_tries


def remove_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
    adjacency, n, rng, max_tries = _input_checks(
        adjacency, random_seed, effect_size, max_tries
    )

    row_inds, col_inds = np.nonzero(adjacency)
    select_edges = rng.choice(len(row_inds), size=effect_size, replace=False)
    select_row_inds = row_inds[select_edges]
    select_col_inds = col_inds[select_edges]
    adjacency[select_row_inds, select_col_inds] = 0

    return adjacency


# TODO
# @jit(nopython=True)
# https://numba-how-to.readthedocs.io/en/latest/numpy.html
def add_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
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

    if tries == max_tries and effect_size != 0:
        msg = (
            "Maximum number of tries reached when adding edges, number added was"
            " less than specified."
        )
        raise UserWarning(msg)

    return adjacency


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


