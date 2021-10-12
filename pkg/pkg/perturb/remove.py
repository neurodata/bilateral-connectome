import numpy as np
from sklearn.utils import check_random_state


def remove_edges(adjacency, n_remove=100, random_state=None):
    random_state = check_random_state(random_state)
    nonzero_inds = np.nonzero(adjacency)
    edge_counter = np.range(len(nonzero_inds[0]))
    random_state.shuffle(edge_counter)
    for i in edge_counter[:n_remove]:
        source = nonzero_inds[0][i]
        target = nonzero_inds[1][i]
        adjacency[source, target] = 0
    return adjacency
