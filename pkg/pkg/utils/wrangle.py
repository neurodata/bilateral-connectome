import numpy as np
from graspologic.utils import largest_connected_component
import pandas as pd


def get_paired_inds(meta, check_in=True, pair_key="pair", pair_id_key="pair_id"):
    pair_meta = meta.copy()
    pair_meta["_inds"] = range(len(pair_meta))

    # remove any center neurons
    pair_meta = pair_meta[pair_meta["hemisphere"].isin(["L", "R"])]

    # remove any neurons for which the other in the pair is not in the metadata
    if check_in:
        pair_meta = pair_meta[pair_meta[pair_key].isin(pair_meta.index)]

    # remove any pairs for which there is only one neuron
    pair_group_size = pair_meta.groupby(pair_id_key).size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta[pair_id_key].isin(remove_pairs)]

    # make sure each pair is "valid" now
    assert pair_meta.groupby(pair_id_key).size().min() == 2
    assert pair_meta.groupby(pair_id_key).size().max() == 2

    # sort into pairs interleaved
    pair_meta.sort_values([pair_id_key, "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["_inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["_inds"]

    # double check that everything worked
    assert (
        meta.iloc[lp_inds][pair_id_key].values == meta.iloc[rp_inds][pair_id_key].values
    ).all()
    return lp_inds, rp_inds


def get_paired_subgraphs(adj, lp_inds, rp_inds):
    ll_adj = adj[np.ix_(lp_inds, lp_inds)]
    rr_adj = adj[np.ix_(rp_inds, rp_inds)]
    lr_adj = adj[np.ix_(lp_inds, rp_inds)]
    rl_adj = adj[np.ix_(rp_inds, lp_inds)]
    return (ll_adj, rr_adj, lr_adj, rl_adj)


def to_largest_connected_component(adj, meta=None):
    adj, lcc_inds = largest_connected_component(adj, return_inds=True)
    if meta is not None:
        return adj, meta.iloc[lcc_inds]
    else:
        return adj


def to_pandas_edgelist(g):
    """Works for multigraphs, the networkx one wasnt returning edge keys"""
    rows = []
    for u, v, k in g.edges(keys=True):
        data = g.edges[u, v, k]
        data["source"] = u
        data["target"] = v
        data["key"] = k
        rows.append(data)
    edges = pd.DataFrame(rows)
    edges["edge"] = list(zip(edges["source"], edges["target"], edges["key"]))
    edges.set_index("edge", inplace=True)
    return edges


def get_paired_nodes(nodes):
    paired_nodes = nodes[nodes["pair_id"] != -1]
    pair_ids = paired_nodes["pair_id"]

    pair_counts = pair_ids.value_counts()
    pair_counts = pair_counts[pair_counts == 1]
    pair_ids = pair_ids[pair_ids.isin(pair_counts.index)]

    paired_nodes = paired_nodes[paired_nodes["pair_id"].isin(pair_ids)].copy()

    return paired_nodes


def get_seeds(left_nodes, right_nodes):
    left_paired_nodes = get_paired_nodes(left_nodes)
    right_paired_nodes = get_paired_nodes(right_nodes)

    pairs_in_both = np.intersect1d(
        left_paired_nodes["pair_id"], right_paired_nodes["pair_id"]
    )
    left_paired_nodes = left_paired_nodes[
        left_paired_nodes["pair_id"].isin(pairs_in_both)
    ]
    right_paired_nodes = right_paired_nodes[
        right_paired_nodes["pair_id"].isin(pairs_in_both)
    ]

    left_seeds = left_paired_nodes.sort_values("pair_id")["inds"]
    right_seeds = right_paired_nodes.sort_values("pair_id")["inds"]

    assert (
        left_nodes.iloc[left_seeds]["pair_id"].values
        == right_nodes.iloc[right_seeds]["pair_id"].values
    ).all()

    return (left_seeds, right_seeds)


def remove_group(
    left_adj, right_adj, left_nodes, right_nodes, group, group_key="simple_group"
):
    left_nodes["inds"] = range(len(left_nodes))
    sub_left_nodes = left_nodes[left_nodes[group_key] != group]
    sub_left_inds = sub_left_nodes["inds"].values
    right_nodes["inds"] = range(len(right_nodes))
    sub_right_nodes = right_nodes[right_nodes[group_key] != group]
    sub_right_inds = sub_right_nodes["inds"].values

    sub_left_adj = left_adj[np.ix_(sub_left_inds, sub_left_inds)]
    sub_right_adj = right_adj[np.ix_(sub_right_inds, sub_right_inds)]
    
    return sub_left_adj, sub_right_adj, sub_left_nodes, sub_right_nodes
