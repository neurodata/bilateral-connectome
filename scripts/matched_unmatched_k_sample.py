#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.perturb import shuffle_edges, add_edges

from graspologic.simulations import er_np
from graspologic.plot import heatmap

p1 = 0.1
n = 50
m_per_group = 5
rng = np.random.default_rng(88)

within_group_shuffle = 100
A1 = er_np(n, p1, directed=True, loops=True)
As = []
for i in range(m_per_group):
    As.append(shuffle_edges(A1, effect_size=within_group_shuffle, random_seed=rng))
As = np.stack(As)

between_group_shuffle = 100
B1 = shuffle_edges(A1, effect_size=between_group_shuffle, random_seed=rng)

between_group_add = 0
if between_group_add > 0:
    B1 = add_edges(B1, effect_size=between_group_add, random_seed=rng)

Bs = []
for i in range(m_per_group):
    Bs.append(shuffle_edges(B1, effect_size=within_group_shuffle, random_seed=rng))
Bs = np.stack(Bs)

#%%

fig, axs = plt.subplots(2, m_per_group, figsize=(3 * m_per_group, 6))

for i, A in enumerate(As):
    heatmap(A, ax=axs[0, i], cbar=False)

for i, B in enumerate(Bs):
    heatmap(B, ax=axs[1, i], cbar=False)

fig.set_facecolor("w")

#%%

label_mask = np.array(m_per_group * [0] + m_per_group * [1])
adjacency_tensor = np.concatenate((As, Bs), axis=0)
adjacency_tensor.shape

#%%
rows, cols = np.indices((n, n))
rows = rows.ravel()
cols = cols.ravel()
groups = np.unique(label_mask)
tables = []
for (i, j) in zip(rows, cols):
    values_at_ij = adjacency_tensor[:, i, j]
    table = np.zeros((2, 2))
    for group in groups:
        values_at_ij[label_mask == group]

#%%
adjacency_tensor[label_mask == 0, :, :].sum(axis=0)
