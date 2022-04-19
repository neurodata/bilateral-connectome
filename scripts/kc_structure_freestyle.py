#%%
from pkg.data import load_unmatched

left_adj, left_nodes = load_unmatched("left")
right_adj, right_nodes = load_unmatched("right")

#%%
left_nodes["class1"].unique()

from pkg.data import load_maggot_graph

mg = load_maggot_graph()
#%%
left_mg = mg.node_subgraph(mg[mg.nodes["left"]].nodes.index)
right_mg = mg.node_subgraph(mg[mg.nodes["right"]].nodes.index)

# %%
nodes = left_mg.nodes
upns_left = nodes[nodes["merge_class"] == "uPN"].index
kcs_left = nodes[(nodes["class1"] == "KC") & (nodes["merge_class"] != "KC-1claw")].index

nodes = right_mg.nodes
upns_right = nodes[nodes["merge_class"] == "uPN"].index
kcs_right = nodes[
    (nodes["class1"] == "KC") & (nodes["merge_class"] != "KC-1claw")
].index

#%%
from graspologic.plot import heatmap
from graspologic.utils import binarize
import numpy as np

left_index = np.concatenate((upns_left, kcs_left))
left_mg.nodes = left_mg.nodes.reindex(left_index)
left_subgraph_mg = left_mg.node_subgraph(upns_left, kcs_left)
left_adj = binarize(left_subgraph_mg.sum.adj)
heatmap(left_adj, cbar=False)

right_index = np.concatenate((upns_right, kcs_right))
right_mg.nodes = right_mg.nodes.reindex(right_index)
right_subgraph_mg = right_mg.node_subgraph(upns_right, kcs_right)
right_adj = binarize(right_subgraph_mg.sum.adj)
heatmap(right_adj, cbar=False)

#%%
print(len(upns_left))
print("vs")
print(len(upns_right))
print()
print(len(kcs_left))
print("vs")
print(len(kcs_right))

#%%
from graspologic.match import GraphMatch

n_upns = len(upns_left)

gm = GraphMatch(n_init=50)

gm.fit(left_adj, right_adj, seeds_A=np.arange(n_upns), seeds_B=np.arange(n_upns))

#%%
import matplotlib.pyplot as plt
from giskard.plot import matrixplot
from pkg.plot import set_theme

set_theme()

perm_inds = gm.perm_inds_[: len(left_adj)]

right_adj_perm = right_adj[perm_inds][:, perm_inds]

diff = left_adj - right_adj_perm
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

matrixplot(left_adj[:n_upns, n_upns:], cbar=False, ax=axs[0])
matrixplot(right_adj_perm[:n_upns, n_upns:], cbar=False, ax=axs[1])
matrixplot(diff[:n_upns, n_upns:], cbar=False, ax=axs[2])

# axs[0].set_title("KCs (graph matched)")
axs[0].set_ylabel("Left", rotation=0, ha="right")
axs[1].set_ylabel("Right", rotation=0, ha="right")
axs[2].set_ylabel("L - R", rotation=0, ha="right")

fig.set_facecolor("w")
#%%


def compute_density(adjacency, loops=False):
    if not loops:
        triu_inds = np.triu_indices_from(adjacency, k=1)
        tril_inds = np.tril_indices_from(adjacency, k=-1)
        n_edges = np.count_nonzero(adjacency[triu_inds]) + np.count_nonzero(
            adjacency[tril_inds]
        )
    else:
        n_edges = np.count_nonzero(adjacency)
    n_nodes = adjacency.shape[0]
    n_possible = n_nodes**2
    if not loops:
        n_possible -= n_nodes
    return n_edges / n_possible


def compute_alignment_strength(A, B, perm=None):
    n = A.shape[0]
    if perm is not None:
        B_perm = B[perm][:, perm]
    else:
        B_perm = B
    n_disagreements = np.count_nonzero(A - B_perm)  # TODO this assumes loopless
    p_disagreements = n_disagreements / (n**2 - n)
    densityA = compute_density(A)
    densityB = compute_density(B)
    denominator = densityA * (1 - densityB) + densityB * (1 - densityA)
    alignment_strength = 1 - p_disagreements / denominator
    return alignment_strength


def obj_func(A, B, perm):
    PBPT = B[perm[: len(A)]][:, perm[: len(A)]]
    return np.linalg.norm(A - PBPT, ord="fro") ** 2, PBPT


# %%

A = left_adj
B = right_adj

n_init = 25

rows = []

gm = GraphMatch(n_init=n_init)
gm.fit(A, B, seeds_A=np.arange(n_upns), seeds_B=np.arange(n_upns))
perm_inds = gm.perm_inds_
score, B_perm = obj_func(A, B, perm_inds)
alignment = compute_alignment_strength(A, B_perm)

rows.append({"data": "Observed", "score": score, "alignment": alignment})


from graspologic.simulations import er_np
from tqdm import tqdm
import pandas as pd


def get_subgraph(A):
    return A[:n_upns, n_upns:]


def make_subgraph(A):
    A = A.copy()
    # all KC to upn
    A[n_upns:] = 0
    # all PN to PN
    A[:n_upns, :n_upns] = 0
    return A


def er_subgraph(size, p, rng=None):
    subgraph = rng.binomial(1, p, size=size)
    n = size[0] + size[1]
    A = np.zeros((n, n))
    A[:n_upns, n_upns:] = subgraph
    return A


left_sub_adj = get_subgraph(A)
right_sub_adj = get_subgraph(B)

p_left = np.count_nonzero(left_sub_adj) / left_sub_adj.size
p_right = np.count_nonzero(right_sub_adj) / right_sub_adj.size
# p = (p_left + p_right) / 2
n = left_sub_adj.shape[1]

rng = np.random.default_rng(888)
n_sims = 100

for sim in tqdm(range(n_sims)):
    A_sim = er_subgraph(left_sub_adj.shape, p_left, rng)
    B_sim = er_subgraph(right_sub_adj.shape, p_right, rng)

    gm = GraphMatch(n_init=n_init)
    gm.fit(A_sim, B_sim, seeds_A=np.arange(n_upns), seeds_B=np.arange(n_upns))
    perm_inds = gm.perm_inds_
    score, B_sim_perm = obj_func(A_sim, B_sim, perm_inds)
    alignment = compute_alignment_strength(A_sim, B_sim_perm)

    rows.append({"data": "ER", "score": score, "alignment": alignment})

results = pd.DataFrame(rows)

#%%
from giskard.plot import histplot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
histplot(data=results, x="alignment", hue="data", kde=False, ax=ax)
ax.set(ylabel="", yticks=[], xlabel="Alignment strength")
ax.spines["left"].set_visible(False)


#%%
heatmap(A_sim)
heatmap(A)
print(np.count_nonzero(A_sim))
print(np.count_nonzero(A))

# %%
