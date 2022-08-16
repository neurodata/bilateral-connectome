#%%
import numpy as np
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt

np.random.seed(88887)
Q = special_ortho_group.rvs(2)
X = np.random.uniform(0, 1, (20, 2))
Y = X @ Q
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.axvline(0, color="k")
ax.axhline(0, color="k")
ax.scatter(X[:, 0], X[:, 1], color="lightblue")
ax.scatter(Y[:, 0], Y[:, 1], color="red")

rads_start = 0
rads_end = np.arctan2(Q[1, 0], Q[0, 0])

Zs = []
for rads in np.linspace(rads_start, rads_end, 100):
    semi_Q = np.array(
        [
            [np.cos(rads), -np.sin(rads)],
            [np.sin(rads), np.cos(rads)],
        ]
    )
    Z = X @ semi_Q
    Zs.append(Z)
Zs = np.stack(Zs)

ax.plot(Zs[:, :, 0], Zs[:, :, 1], color="lightgrey", zorder=-1)

ax.axis("off")

#%%
