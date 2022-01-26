import numpy as np
from numpy.random import default_rng
from giskard.utils import get_random_seed
from sklearn.utils import shuffle


def _input_checks(adjacency, random_seed, effect_size, max_tries):
    adjacency = adjacency.copy()

    if isinstance(random_seed, np.integer) or random_seed is None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    if max_tries is None:
        max_tries = effect_size * 10

    return adjacency, rng, max_tries


def remove_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
    n_nonzero = np.count_nonzero(adjacency)
    if effect_size > n_nonzero:
        return None

    adjacency, rng, max_tries = _input_checks(
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

# TODO add an induced option
# TODO deal with max number of edges properly
# TODO put down edges all at once rather than this silly thing
def add_edges(adjacency, effect_size=100, random_seed=None, max_tries=None):
    adjacency, rng, max_tries = _input_checks(
        adjacency, random_seed, effect_size, max_tries
    )

    n_source = adjacency.shape[0]
    n_target = adjacency.shape[1]
    n_possible = n_source * n_target
    if effect_size > n_possible:  # technicall should be - n if on main diagonal
        return

    n_edges_added = 0
    tries = 0
    while n_edges_added < effect_size and tries < max_tries:
        i = rng.integers(n_source)
        j = rng.integers(n_target)
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

    if adjacency is None:
        return

    seed = get_random_seed(rng)
    adjacency = add_edges(
        adjacency, effect_size=effect_size, random_seed=seed, max_tries=max_tries
    )

    return adjacency


def perturb_subgraph(adjacency, perturb_func, source_nodes, target_nodes, **kwargs):
    adjacency = adjacency.copy()
    A_subgraph = adjacency[source_nodes][:, target_nodes]
    A_subgraph_perturbed = perturb_func(A_subgraph, **kwargs)
    if A_subgraph_perturbed is None:
        return None
    # ix_ seems necessary here for assignment into
    adjacency[np.ix_(source_nodes, target_nodes)] = A_subgraph_perturbed
    return adjacency


def shuffle_edges_subgraph(
    adjacency,
    source_nodes,
    target_nodes,
    effect_size=100,
    random_seed=None,
    max_tries=None,
):
    return perturb_subgraph(
        adjacency,
        shuffle_edges,
        source_nodes,
        target_nodes,
        effect_size=effect_size,
        random_seed=random_seed,
        max_tries=max_tries,
    )


def remove_edges_subgraph(
    adjacency,
    source_nodes,
    target_nodes,
    effect_size=100,
    random_seed=None,
    max_tries=None,
):
    return perturb_subgraph(
        adjacency,
        remove_edges,
        source_nodes,
        target_nodes,
        effect_size=effect_size,
        random_seed=random_seed,
        max_tries=max_tries,
    )


def add_edges_subgraph(
    adjacency,
    source_nodes,
    target_nodes,
    effect_size=100,
    random_seed=None,
    max_tries=None,
):
    return perturb_subgraph(
        adjacency,
        add_edges,
        source_nodes,
        target_nodes,
        effect_size=effect_size,
        random_seed=random_seed,
        max_tries=max_tries,
    )
