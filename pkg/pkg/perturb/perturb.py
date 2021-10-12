import numpy as np
from sklearn.utils import check_random_state


def perturb(adjacency, effect_size=100, mode="add"):
    # TODO type checking, etc
    pass


def remove_edges(adjacency, effect_size=100, random_state=None):
    random_state = check_random_state(random_state)
    adjacency = adjacency.copy()

    nonzero_inds = np.nonzero(adjacency)
    edge_counter = np.arange(len(nonzero_inds[0]))
    random_state.shuffle(edge_counter)
    for i in edge_counter[:effect_size]:
        source = nonzero_inds[0][i]
        target = nonzero_inds[1][i]
        adjacency[source, target] = 0
    return adjacency


def swap_edges(adjacency, effect_size=100, random_state=None):
    return 0


def add_edges(adjacency, effect_size=100, random_state=None):
    return 0
