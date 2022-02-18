import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.simulations import sbm


def sample_toy_networks(seed=888888, ns=None, B=None):
    np.random.seed(seed)
    if ns is None:
        ns = [5, 6, 7]
    if B is None:
        B = np.array([[0.8, 0.2, 0.05], [0.05, 0.9, 0.2], [0.05, 0.05, 0.7]])
    A1, labels = sbm(ns, B, directed=True, loops=False, return_labels=True)
    A2 = sbm(ns, B, directed=True, loops=False)

    node_data = pd.DataFrame(index=np.arange(A1.shape[0]))
    node_data["labels"] = labels + 1
    return A1, A2, node_data


def get_toy_palette():
    return dict(zip([1, 2, 3], sns.color_palette("Set2")[3:]))
