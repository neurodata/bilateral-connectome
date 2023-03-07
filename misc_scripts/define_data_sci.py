#%% [markdown]
# # Larval *Drosophila melanogaster* brain connectome
#
# Recently, authors mapped a connectome of a *Drosophila melanogaster* larva. This
# synaptic wiring diagram is comprised of 3,013 neurons and over 544,000
# synapses. Importantly, this work yielded a complete reconstruction of both the left
# and right hemispheres of the brain. We can represent this data as a network, with
# nodes representing neurons and
# edges representing some number of synapses between them.
# Since there are many modeling choices to make even when deciding how to take a raw
# connectome and generate a network representation from it, we describe some of these
# choices below.

#%% [markdown]
# ## Unmatched data
# First we consider a left/right hemisphere dataset which does not require any
# neuron-to-neuron correspondence between the two hemispheres.
#
# We remove all neurons who do not have at least 3 inputs and at least 3 outputs in the
# brain network - these are mostly "young" neurons. We then take the largest connected
# component after this removal.

# Then, for the networks we are going to compare, we select only the left-to-left
# (ipsilateral) induced subgraph, and likewise for the right-to-right. Note that there
# are conceivable ways to define a notion of bilateral symmetry which does include the
# contralateral connections as well, but we do not consider that here.

# For the networks themselves, we chose to operate on networks which are:
# - **Unweighted**: we do not consider the number of synapses between two neurons in the
#   current analysis. For the current network of
#   interest, four edge types are available: axo-axonic, axo-dendritic,
#   dendro-dendritic, and dendro-axonic. We make no distinction between these four edges
#   types. For now, we consider there to be an edge if there is at least one
#   synapse between any two neurons (of any edge type).  One could consider notions of
#   bilateral symmetry
#   for a weighted network, and even for the unweighted case, one could consider varying
#   thresholds of the network based on varying edge weights (which have often been
#   employed in connectomics studies). For now, we focus on the unweighted case purely
#   for simplicity and the availability of more two-sample tests for unweighted
#   networks,
#   though the weighted case or choice of threshold are also of great interest.
# - **Directed**: we allow for a distinction between edges which go from neuron $i$ to
#   neuron $j$ as opposed to from $j$ to $i$.
# - **Loopless**: we remove any edges which go from neuron $i$ to neuron $i$.
#%%

import datetime
import time

import networkx as nx
import numpy as np
import pandas as pd
from pkg.data import DATA_PATH, DATA_VERSION
from pkg.io import get_environment_variables
from pkg.io import glue as default_glue

from graspologic.utils import is_fully_connected, remove_loops

t0 = time.time()

FILENAME = "define_data"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, display=False, **kwargs)


RESAVE_DATA, _, _ = get_environment_variables()

print(f"Using data from {DATA_VERSION}")
output_dir = DATA_PATH / "processed-science"
input_dir = DATA_PATH / DATA_VERSION

#%%
adjacency = pd.read_csv(
    input_dir / "Supplementary-Data-S1" / "all-all_connectivity_matrix.csv", index_col=0
)
adjacency.columns = adjacency.columns.astype(int)
adjacency = pd.DataFrame(
    data=adjacency.values, index=adjacency.index, columns=adjacency.columns
)
inputs = pd.read_csv(input_dir / "Supplementary-Data-S1" / "inputs.csv", index_col=0)
outputs = pd.read_csv(input_dir / "Supplementary-Data-S1" / "outputs.csv", index_col=0)

#%%
meta = pd.read_csv(input_dir / "Supplementary_Data_S2.csv")
meta = meta.replace("no pair", np.nan)

meta = meta.melt(
    id_vars=["celltype", "additional_annotations", "level_7_cluster"],
    value_vars=["left_id", "right_id"],
    value_name="id",
    var_name="hemisphere",
).dropna()

meta["hemisphere"] = meta["hemisphere"].map({"left_id": "L", "right_id": "R"})

meta["id"] = meta["id"].astype(int)
meta = meta.set_index("id").sort_index()

#%%
meta.loc[meta.index.duplicated(False)]

#%%
meta = meta.loc[~meta.index.duplicated()]

#%%

meta["celltype"].unique()

#%%
# updating to match the nomenclature in the figures in Winding/Pedigo et al 2023 Science
celltype_map = {
    "KC": "KC",
    "sensory": "Sensory",
    "DN-SEZ": r"DN$^{SEZ}$",
    "PN": "Projection",
    "ascending": "Ascending",
    "DN-VNC": r"DN$^{VNC}$",
    "PN-somato": r"Projection$^{somato}$",
    "pre-DN-VNC": r"pre-DN$^{VNC}$",
    "other": "Other",
    "pre-DN-SEZ": r"pre-DN$^{SEZ}$",
    "RGN": "RGN",
    "LHN": "Lateral Horn",
    "MBIN": "MBIN",
    "CN": "CN",
    "LN": "Local",
    "MB-FBN": "MB-FBN",
    "MBON": "MBON",
    "MB-FFN": "MB-FFN",
    "MBIN-FFN": "MBIN-FFN",
    "MBIN-FBN": "MBIN-FBN",
}

meta["celltype_discrete"] = meta["celltype"].map(celltype_map)

#%%

assert is_fully_connected(adjacency.values)

#%%
meta.loc[meta.index.duplicated(False)]

#%%
intersect_index = adjacency.index.intersection(meta.index)
intersect_index.name = "id"
adjacency = adjacency.loc[intersect_index, intersect_index]
meta = meta.loc[intersect_index]
g = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)

if RESAVE_DATA:
    print("Saved full data")
    nx.write_edgelist(
        g,
        output_dir / "unmatched_full_edgelist.csv",
        delimiter=",",
        data=["weight"],
    )
    meta.to_csv(output_dir / "unmatched_full_nodes.csv")


#%%
meta = meta.sort_values(["hemisphere", "celltype_discrete"])

left_nodes = meta.query("hemisphere == 'L'")
right_nodes = meta.query("hemisphere == 'R'")

left_adjacency = adjacency.loc[left_nodes.index, left_nodes.index]
right_adjacency = adjacency.loc[right_nodes.index, right_nodes.index]

#%%
assert is_fully_connected(left_adjacency.values)
assert is_fully_connected(right_adjacency.values)

#%%
left_adj = left_adjacency.values
right_adj = right_adjacency.values

#%%
left_n_loops = np.count_nonzero(np.diag(left_adj))
right_n_loops = np.count_nonzero(np.diag(right_adj))
left_n_edges = np.count_nonzero(left_adj)
right_n_edges = np.count_nonzero(right_adj)
p_loops = (left_n_loops + right_n_loops) / (len(left_adj) + len(right_adj))
p_loop_edges = (left_n_loops + right_n_loops) / (left_n_edges + right_n_edges)
glue("p_loops", p_loops, form="2.0f%")
glue("p_loops_edges", p_loop_edges, form=".1f%")

#%%
left_adj = remove_loops(left_adj)
right_adj = remove_loops(right_adj)

n_left_unmatched = left_adj.shape[0]
n_right_unmatched = right_adj.shape[0]
glue("n_left_unmatched", n_left_unmatched, form="long")
glue("n_right_unmatched", n_right_unmatched, form="long")


#%% [markdown]
# After this data cleaning, we are left with {glue:text}`n_left_unmatched` neurons in
# the left
# hemisphere, and {glue:text}`n_right_unmatched` neurons in the right hemisphere.

# %%
left_adj = pd.DataFrame(
    data=left_adj.astype(int), index=left_nodes.index, columns=left_nodes.index
)
left_g = nx.from_pandas_adjacency(left_adj, create_using=nx.DiGraph)

if RESAVE_DATA:
    nx.write_edgelist(
        left_g,
        output_dir / "unmatched_left_edgelist.csv",
        delimiter=",",
        data=["weight"],
    )
    left_nodes.to_csv(output_dir / "unmatched_left_nodes.csv")


right_adj = pd.DataFrame(
    data=right_adj.astype(int), index=right_nodes.index, columns=right_nodes.index
)
right_g = nx.from_pandas_adjacency(right_adj, create_using=nx.DiGraph)

if RESAVE_DATA:
    nx.write_edgelist(
        right_g,
        output_dir / "unmatched_right_edgelist.csv",
        delimiter=",",
        data=["weight"],
    )
    right_nodes.to_csv(output_dir / "unmatched_right_nodes.csv")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
