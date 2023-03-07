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
from pkg.utils import set_warnings

set_warnings()

import datetime
import os
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from graspologic.utils import remove_loops
from pkg.io import glue as default_glue
from pkg.io import get_environment_variables
from pkg.data import DATA_VERSION, load_maggot_graph, select_nice_nodes

t0 = time.time()

FILENAME = "define_data"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, display=False, **kwargs)


RESAVE_DATA, _, _ = get_environment_variables()

print(f"Using data from {DATA_VERSION}")
os.chdir("/Users/bpedigo/JHU_code/bilateral")  # TODO fix, make this less fragile
output_dir = os.path.join(os.getcwd(), "bilateral-connectome/data/processed-2022-11-03")
output_dir = Path(output_dir)

#%%
mg = load_maggot_graph()

print(f"Before anything: {len(mg)}")
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
total_paper_neurons = len(mg)
glue("total_paper_neurons", total_paper_neurons, form="long")
print(f"After filter by groups to consider for paper: {len(mg)}")

#%%
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
print(f"After removing non left/right: {len(mg)}")
mg.to_largest_connected_component(verbose=False)
print(f"After largest connected component: {len(mg)}")

mg.nodes.sort_values("hemisphere", inplace=True)
mg.nodes["_inds"] = range(len(mg.nodes))

mg.nodes["celltype_discrete"] = mg.nodes["simple_group"]

left_mg, right_mg = mg.bisect(lcc=True)
left_nodes = left_mg.nodes
right_nodes = right_mg.nodes
print(
    f"After taking largest connected component of each side {len(left_nodes) + len(right_nodes)}"
)

left_adj = left_mg.sum.adj
right_adj = right_mg.sum.adj


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


left_adj = remove_loops(left_adj)
right_adj = remove_loops(right_adj)

n_left_matched = left_adj.shape[0]
n_right_matched = right_adj.shape[0]

glue("n_left_matched", n_left_matched)
glue("n_right_matched", n_right_matched)

#%% [markdown]
# For the matched networks, we are left with {glue:text}`n_left_matched` neurons in the
# left
# hemisphere, and {glue:text}`n_right_matched` neurons in the right hemisphere.

# %%
left_adj = pd.DataFrame(
    data=left_adj.astype(int), index=left_nodes.index, columns=left_nodes.index
)
left_g = nx.from_pandas_adjacency(left_adj, create_using=nx.DiGraph)
if RESAVE_DATA:
    nx.write_edgelist(
        left_g,
        output_dir / "matched_left_edgelist.csv",
        delimiter=",",
        data=["weight"],
    )

    left_nodes.to_csv(output_dir / "matched_left_nodes.csv")


right_adj = pd.DataFrame(
    data=right_adj.astype(int), index=right_nodes.index, columns=right_nodes.index
)
right_g = nx.from_pandas_adjacency(right_adj, create_using=nx.DiGraph)

if RESAVE_DATA:
    nx.write_edgelist(
        right_g,
        output_dir / "matched_right_edgelist.csv",
        delimiter=",",
        data=["weight"],
    )
    right_nodes.to_csv(output_dir / "matched_right_nodes.csv")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
