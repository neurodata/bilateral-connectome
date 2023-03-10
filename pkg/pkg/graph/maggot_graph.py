from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd

from graspologic.utils import largest_connected_component

from .utils import get_paired_inds, to_pandas_edgelist


class MaggotGraph:
    def __init__(self, g, nodes=None, edges=None):
        self.g = g
        # TODO add checks for when nodes/edges are passed, do they actually match the
        # graph?
        if nodes is None:
            # TODO
            raise NotImplementedError()
        self.nodes = nodes
        _add_missing_nodes(g, nodes.index)
        if edges is None:
            edges = to_pandas_edgelist(g)
        self.edges = edges
        self._node_columns = nodes.columns

    def to_edge_type_graph(self, edge_type):
        type_edges = self.edges[self.edges["edge_type"] == edge_type]
        view = nx.edge_subgraph(self.g, type_edges.index).copy()
        return MaggotGraph(view, self.nodes, type_edges)

    @property
    def edge_types(self):
        return sorted(self.edges["edge_type"].unique())

    @property
    def is_single_type(self):
        return len(self.edge_types) == 1

    @property
    def aa(self):
        return self.to_edge_type_graph("aa")

    @property
    def ad(self):
        return self.to_edge_type_graph("ad")

    @property
    def da(self):
        return self.to_edge_type_graph("da")

    @property
    def dd(self):
        return self.to_edge_type_graph("dd")

    @property
    def sum(self):
        return self.to_edge_type_graph("sum")

    @property
    def adj(self):
        if self.is_single_type:
            adj = nx.to_numpy_array(self.g, nodelist=self.nodes.index)
            return adj
        else:
            msg = "Current MaggotGraph has more than one edge type. "
            msg += "Use .adjs() method instead to specify multple edge types."
            raise ValueError(msg)

    @property
    def adjs(self):
        adjs = []
        for edge_type in self.edge_types:
            adj = self.to_edge_type_graph(edge_type).adj
            adjs.append(adj)
        adjs = np.stack(adjs)
        return adjs

    def node_subgraph(self, source_node_ids, target_node_ids=None):
        # if target_node_ids is None:  # induced subgraph on source nodes
        #     # TODO don't really need two cases here
        #     sub_g = self.g.subgraph(source_node_ids)
        #     sub_nodes = self.nodes.reindex(source_node_ids)
        #     sub_edges = to_pandas_edgelist(sub_g)
        #     return MaggotGraph(sub_g, sub_nodes, sub_edges)
        # else:  # subgraph defined on a set of nodes, but not necessarily induced
        induced = False
        if target_node_ids is None:
            target_node_ids = source_node_ids
            induced = True
        edges = self.edges
        nodes = self.nodes
        source_edges = edges[edges.source.isin(source_node_ids)]
        source_target_edges = source_edges[source_edges.target.isin(target_node_ids)]
        sub_g = self.g.edge_subgraph(source_target_edges.index).copy()
        sub_nodes = nodes[
            nodes.index.isin(source_node_ids) | nodes.index.isin(target_node_ids)
        ]
        if induced:
            sub_nodes = sub_nodes.reindex(source_node_ids)
        # TODO what ordering makes sense when the subgraph is not induced
        return MaggotGraph(sub_g, sub_nodes, source_target_edges)

    def copy(self):
        return deepcopy(self)

    def __len__(self):
        return len(self.g)

    def __repr__(self):
        return self.summary_statistics.__repr__()

    def _repr_html_(self):
        return self.summary_statistics._repr_html_()

    @property
    def summary_statistics(self):
        edge_types = self.edge_types
        edges = self.edges
        cols = []
        for edge_type in edge_types:
            type_edges = edges[edges["edge_type"] == edge_type]
            # number of actual nodes being used (ignoring edgeless ones)
            n_nodes = len(np.unique(type_edges[["source", "target"]].values.ravel()))
            n_edges = len(type_edges)
            edgesum = type_edges["weight"].sum()
            data = [n_nodes, n_edges, edgesum]
            index = ["n_nodes", "n_edges", "sum_edge_weights"]
            cols.append(pd.Series(index=index, data=data, name=edge_type))
        results = pd.DataFrame(cols)
        results.index.name = "edge_type"
        return results

    def __getitem__(self, key):
        if isinstance(key, pd.Series) and key.dtype == bool:
            return self.node_subgraph(key[key].index)

    def __setitem__(self, key, val):
        self.nodes[key] = val

    def bisect(
        self,
        paired=False,
        lcc=False,
        check_in=True,
        pair_key="pair",
        pair_id_key="pair_id",
    ):
        """[summary]

        Parameters
        ----------
        paired : bool, optional
            If ``paired``, return subgraphs only for paired neurons and indexed the same
            for left and right. Otherwise, return subgraphs in any order, and for all
            left/right neurons.

        Raises
        ------
        NotImplementedError
            [description]
        """
        nodes = self.nodes
        if paired:
            lp_inds, rp_inds = get_paired_inds(
                nodes,
                check_in=check_in,
                pair_key=pair_key,
                pair_id_key=pair_id_key,
            )
            left_ids = nodes.iloc[lp_inds].index
            right_ids = nodes.iloc[rp_inds].index
        else:
            left_ids = nodes[nodes["hemisphere"] == "L"].index
            right_ids = nodes[nodes["hemisphere"] == "R"].index
        left_left_mg = self.node_subgraph(left_ids)
        right_right_mg = self.node_subgraph(right_ids)
        left_left_mg = self.node_subgraph(left_ids)

        if lcc:
            left_left_mg.to_largest_connected_component()
            right_right_mg.to_largest_connected_component()

        if paired and lcc:
            # when taking the LCCs in the above, individual graphs can be disconnected,
            # so this repeats the process to make sure we are on the right node set
            index = np.concatenate(
                (left_left_mg.nodes.index.values, right_right_mg.nodes.index.values)
            )
            subgraph = self.node_subgraph(index)
            left_left_mg, right_right_mg = subgraph.bisect(
                paired=paired,
                lcc=False,
                check_in=check_in,
                pair_key=pair_key,
                pair_id_key=pair_id_key,
            )
            assert len(left_left_mg) == len(right_right_mg)

        return left_left_mg, right_right_mg

    def fix_pairs(self, pair_key="pair", pair_id_key="pair_id"):
        nodes = self.nodes
        for node_id, row in nodes.iterrows():
            pair = row[pair_key]
            if pair != -1:
                if pair not in nodes.index:
                    row[pair_key] = -1
                    row[pair_id_key] = -1
                    print(f"Removing invalid pair: {node_id} to {pair}")

    def to_largest_connected_component(self, verbose=False):
        if self.is_single_type:
            adj = self.adj
        else:
            adj = self.adjs.sum(axis=0)
        lcc, inds = largest_connected_component(adj, return_inds=True)
        lcc_node_ids = self.nodes.index[inds]
        remove_node_ids = np.setdiff1d(self.nodes.index, lcc_node_ids)
        self.g.remove_nodes_from(remove_node_ids)
        self.nodes = self.nodes.iloc[inds]
        edges = self.edges
        self.edges = edges[
            edges["source"].isin(lcc_node_ids) & edges["target"].isin(lcc_node_ids)
        ]
        if verbose > 0:
            n_removed = len(remove_node_ids)
            print(
                f"Removed {n_removed} nodes when taking the largest connected component."
            )


def _add_missing_nodes(g, nodes_to_check):
    nodelist = list(g.nodes)
    missing_nodes = np.setdiff1d(nodes_to_check, nodelist)
    g.add_nodes_from(missing_nodes)
