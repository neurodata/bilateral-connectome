#%% [markdown]
# # Defining the data
# Here we describe in more detail what data we will use for the bilateral symmetry
# comparisons.

#%% [markdown]
# ## Unmatched data
# First we consider a left/right hemisphere dataset which does not require any
# neuron-to-neuron correspondence between the two hemispheres.
#
# We remove all neurons who do not have at least 3 inputs and at least 3 outputs in the
# brain network - these are mostly "young" neurons. We then take the largest connected
# component after this removal.
#
# Then, for the networks we are going to compare, we select only the left-to-left
# (ipsilateral) induced subgraph, and likewise for the right-to-right. Note that there
# are conceivable ways to define a notion of bilateral symmetry which does include the
# contralateral connections as well, but we do not consider that here.
#
# For the networks themselves, we chose to operate on networks which are:
# - **Unweighted**: we do not consider the number of synapses between two neurons in the
#   current analysis. For the current network of
#   interest, four edge types are available: axo-axonic, axo-dendritic,
#   dendro-dendritic, and dendro-axonic. We make no distinction between these four edges
#   types. For now, we condider there to be an edge if there is at least one
#   synapse between any two neurons (of any edge type).  One could consider notions of
#   bilateral symmetry
#   for a weighted network, and even for the unweighted case, one could consider varying
#   thresholds of the network based on varying edge weights (which have often been
#   employeed in connectomics studies). For now, we focus on the unweighted case purely
#   for simplicity and the availability of more two-sample tests for unweighted
#   networks,
#   though the weighted case or choice of threshold are also of great interest.
# - **Directed**: we allow for a distinction between edges which go from neuron $i$ to
#   neuron $j$ as opposed to from $j$ to $i$.
# - **Loopless**: we remove any edges which go from neuron $i$ to neuron $i$.
#%%
from pkg.utils import set_warnings

set_warnings()

import datetime
import os
import time
from pathlib import Path

import networkx as nx
import pandas as pd
from graspologic.utils import binarize, remove_loops
from myst_nb import glue
from pkg.data import DATA_VERSION, load_maggot_graph, select_nice_nodes

t0 = time.time()

# print(f"Using data from {DATA_VERSION}")
os.chdir("/Users/bpedigo/JHU_code/bilateral")  # TODO fix, make this less fragile
output_dir = os.path.join(os.getcwd(), "bilateral-connectome/data/processed")
output_dir = Path(output_dir)
# print(f"Saving data to {output_dir}")

#%%
mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True)
left_nodes = left_mg.nodes
right_nodes = right_mg.nodes

left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj

left_adj = binarize(left_adj)
right_adj = binarize(right_adj)

left_adj = remove_loops(left_adj)
right_adj = remove_loops(right_adj)

n_left_unmatched = left_adj.shape[0]
n_right_unmatched = right_adj.shape[0]
glue("n_left_unmatched", n_left_unmatched, display=False)
glue("n_right_unmatched", n_right_unmatched, display=False)

#%% [markdown]
# After this data cleaning, we are left with {glue:text}`n_left_unmatched` neurons in
# the left
# hemisphere, and {glue:text}`n_right_unmatched` neurons in the right hemisphere.

# %%
left_adj = pd.DataFrame(data=left_adj, index=left_nodes.index, columns=left_nodes.index)
left_g = nx.from_pandas_adjacency(left_adj, create_using=nx.DiGraph)
nx.write_edgelist(
    left_g, output_dir / "unmatched_left_edgelist.csv", delimiter=",", data=False
)

left_nodes.to_csv(output_dir / "unmatched_left_nodes.csv")


right_adj = pd.DataFrame(
    data=right_adj, index=right_nodes.index, columns=right_nodes.index
)
right_g = nx.from_pandas_adjacency(right_adj, create_using=nx.DiGraph)
nx.write_edgelist(
    right_g, output_dir / "unmatched_right_edgelist.csv", delimiter=",", data=False
)

right_nodes.to_csv(output_dir / "unmatched_right_nodes.csv")


#%% [markdown]
# ## Matched data
# Next, we consider a left/right hemisphere dataset where we require each neuron in the
# left hemisphere to be matched with a neuron in the right hemisphere. These pairings
# were determined by matching both connectivity and morphology in previous publications.
#
# Not all neurons in the original dataset were matched, so there will be slightly fewer
# neurons in this version of the data. Since we have this matched requirement, both
# networks will necessarily be of the same size.
#
# All other aspects of the networks (unweighted, directed, no loops) are the same as
# described above for the unmatched networks.
#%%
mg = load_maggot_graph()
mg = select_nice_nodes(mg)
left_mg, right_mg = mg.bisect(lcc=True, paired=True)
left_nodes = left_mg.nodes
right_nodes = right_mg.nodes

left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj

left_adj = binarize(left_adj)
right_adj = binarize(right_adj)

left_adj = remove_loops(left_adj)
right_adj = remove_loops(right_adj)

n_left_matched = left_adj.shape[0]
n_right_matched = right_adj.shape[0]

glue("n_left_matched", n_left_matched, display=False)
glue("n_right_matched", n_right_matched, display=False)

#%% [markdown]
# For the matched networks, we are left with {glue:text}`n_left_matched` neurons in the
# left
# hemisphere, and {glue:text}`n_right_matched` neurons in the right hemisphere.

# %%
left_adj = pd.DataFrame(data=left_adj, index=left_nodes.index, columns=left_nodes.index)
left_g = nx.from_pandas_adjacency(left_adj, create_using=nx.DiGraph)
nx.write_edgelist(
    left_g, output_dir / "matched_left_edgelist.csv", delimiter=",", data=False
)

left_nodes.to_csv(output_dir / "matched_left_nodes.csv")


right_adj = pd.DataFrame(
    data=right_adj, index=right_nodes.index, columns=right_nodes.index
)
right_g = nx.from_pandas_adjacency(right_adj, create_using=nx.DiGraph)
nx.write_edgelist(
    right_g, output_dir / "matched_right_edgelist.csv", delimiter=",", data=False
)

right_nodes.to_csv(output_dir / "matched_right_nodes.csv")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
# print(f"Script took {delta}")
# print(f"Completed at {datetime.datetime.now()}")
