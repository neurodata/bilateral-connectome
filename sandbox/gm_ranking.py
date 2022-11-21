from graspologic.utils import remove_loops
import numpy as np
from graspologic.match import graph_match


def signal_flow(A):
    """Implementation of the signal flow metric from Varshney et al 2011

    Parameters
    ----------
    A : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


def rank_signal_flow(A):
    sf = signal_flow(A)
    perm_inds = np.argsort(-sf)
    return perm_inds


def rank_graph_match_flow(A, n_init=10, max_iter=30, eps=1e-4, **kwargs):
    n = len(A)
    try:
        initial_perm = rank_signal_flow(A)
        init = np.eye(n)[initial_perm]
    except np.linalg.LinAlgError:
        print("SVD did not converge in signal flow")
        init = np.full((n, n), 1 / n)
    match_mat = np.zeros((n, n))
    triu_inds = np.triu_indices(n, k=1)
    match_mat[triu_inds] = 1
    gm = GraphMatch(n_init=n_init, max_iter=max_iter, init=init, eps=eps, **kwargs)
    perm_inds = gm.fit_predict(match_mat, A)
    return perm_inds
