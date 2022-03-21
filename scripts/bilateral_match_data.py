#%% [markdown]
# # Matching when including the contralateral connections
#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import get_seeds


import datetime
import time
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from numba import jit

from giskard.plot import matched_stripplot
from pkg.data import load_maggot_graph
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs

from pkg.plot import simple_plot_neurons
from pkg.pymaid import start_instance

from giskard.plot import adjplot, matrixplot
from pkg.io import OUT_PATH
from myst_nb import glue as default_glue

t0 = time.time()

output_path = OUT_PATH / "match_examine"
if not os.path.isdir(output_path):
    raise ValueError("Path for saving output does not exist.")


FILENAME = "bilateral_match_data"

DISPLAY_FIGS = True


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, prefix="fig")

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var, prefix=None):
    savename = f"{FILENAME}-{name}"
    if prefix is not None:
        savename = prefix + ":" + savename
    default_glue(savename, var, display=False)


set_theme()

colors = sns.color_palette("Set1")

#%% [markdown]
# ### Load the data
#%%

from pkg.data import load_maggot_graph, load_matched

from pkg.data import load_network_palette

load_network_palette()

#%%
left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")
left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))
seeds = get_seeds(left_nodes, right_nodes)
all_nodes = pd.concat((left_nodes, right_nodes))
all_nodes["inds"] = range(len(all_nodes))

left_nodes.iloc[seeds[0]]["pair_id"]

#%%
mg = load_maggot_graph()
mg = mg.node_subgraph(all_nodes.index)
adj = mg.sum.adj

#%%
# mg = mg[mg.nodes["paper_clustered_neurons"]]
# mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
# lp_inds, rp_inds = get_paired_inds(mg.nodes)

# # ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect()

# left_in_right = ll_mg.nodes["pair"].isin(rr_mg.nodes.index)
# left_in_right_idx = left_in_right[left_in_right].index
# right_in_left = rr_mg.nodes["pair"].isin(ll_mg.nodes.index)
# right_in_left_idx = right_in_left[right_in_left].index
# left_in_right_pair_ids = ll_mg.nodes.loc[left_in_right_idx, "pair_id"]
# right_in_left_pair_ids = rr_mg.nodes.loc[right_in_left_idx, "pair_id"]
# valid_pair_ids = np.intersect1d(left_in_right_pair_ids, right_in_left_pair_ids)
# n_pairs = len(valid_pair_ids)
# mg.nodes["valid_pair_id"] = False
# mg.nodes.loc[mg.nodes["pair_id"].isin(valid_pair_ids), "valid_pair_id"] = True
# mg.nodes.sort_values(
#     ["hemisphere", "valid_pair_id", "pair_id"], inplace=True, ascending=False
# )
# mg.nodes["_inds"] = range(len(mg.nodes))
# adj = mg.sum.adj
# left_nodes = mg.nodes[mg.nodes["hemisphere"] == "L"].copy()
# left_inds = left_nodes["_inds"]
# right_nodes = mg.nodes[mg.nodes["hemisphere"] == "R"].copy()
# right_inds = right_nodes["_inds"]

max_n_side = max(len(left_nodes), len(right_nodes))

#%%
left_inds = all_nodes.iloc[: len(left_nodes)]["inds"]
right_inds = all_nodes.iloc[len(left_nodes) :]["inds"]


def pad(A, size):
    # naive padding for now
    A_padded = np.zeros((size, size))
    rows = A.shape[0]
    cols = A.shape[1]
    A_padded[:rows, :cols] = A
    return A_padded


ll_adj = pad(adj[np.ix_(left_inds, left_inds)], max_n_side)
rr_adj = pad(adj[np.ix_(right_inds, right_inds)], max_n_side)
lr_adj = pad(adj[np.ix_(left_inds, right_inds)], max_n_side)
rl_adj = pad(adj[np.ix_(right_inds, left_inds)], max_n_side)

for i in range(max_n_side - len(right_inds)):
    right_nodes = right_nodes.append(
        pd.Series(name=-i - 1, dtype="float"), ignore_index=False
    )

# #%%
# plot_kws = dict(plot_type="scattermap", sizes=(1, 1))
# fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw=dict(hspace=0, wspace=0))

# ax = axs[0, 0]
# adjplot(ll_adj, meta=left_nodes, ax=ax, color=palette["Left"], **plot_kws)

# ax = axs[0, 1]
# matrixplot(
#     lr_adj,
#     row_meta=left_nodes,
#     col_meta=right_nodes,
#     ax=ax,
#     color=palette["Contra"],
#     square=True,
#     **plot_kws,
# )

# ax = axs[1, 0]
# matrixplot(
#     rl_adj,
#     col_meta=left_nodes,
#     row_meta=right_nodes,
#     ax=ax,
#     color=palette["Contra"],
#     square=True,
#     **plot_kws,
# )

# ax = axs[1, 1]
# adjplot(rr_adj, meta=right_nodes, ax=ax, color=palette["Right"], **plot_kws)
#%% [markdown]
# ## Include the contralateral connections in graph matching
#%% [markdown]
# ### Run the graph matching experiment
#%%

# n_pairs = len(seeds[0])

from graspologic.match.qap import _doubly_stochastic

np.random.seed(8888)
maxiter = 30
verbose = 1
ot = False
maximize = True
reg = 10  # Sinkhorn
thr = 5e-2  # Sinkhorn
tol = 1e-3  # FW
n_init = 5
alpha = 0.01
n = len(ll_adj)
initialization = "flat"

if initialization == "known":
    pass
    # # construct an initialization
    # P0 = np.zeros((n, n))
    # P0[np.arange(n_pairs), np.arange(n_pairs)] = 1
    # P0[n_pairs:, n_pairs:] = 1 / (n - n_pairs)
elif initialization == "flat":
    P0 = np.full((n, n), 1 / n)


@jit(nopython=True)
def compute_gradient(A, B, AB, BA, P):
    return A @ P @ B.T + A.T @ P @ B + AB @ P.T @ BA.T + BA.T @ P.T @ AB


@jit(nopython=True)
def compute_step_size(A, B, AB, BA, P, Q):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares
    a_cross = np.trace(AB.T @ R @ BA @ R)
    b_cross = np.trace(AB.T @ R @ BA @ Q) + np.trace(AB.T @ Q @ BA @ R)
    a_intra = np.trace(A @ R @ B.T @ R.T)
    b_intra = np.trace(A @ Q @ B.T @ R.T + A @ R @ B.T @ Q.T)

    a = a_cross + a_intra
    b = b_cross + b_intra

    if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
        alpha = -b / (2 * a)
    return alpha
    # else:
    #     alpha = np.argmin([0, (b + a) * obj_func_scalar])
    # return alpha


@jit(nopython=True)
def compute_objective_function(A, B, AB, BA, P):
    return np.trace(A @ P @ B.T @ P.T) + np.trace(AB.T @ P @ BA @ P)


from ot import sinkhorn


def alap(P, reg, maximize=True, tol=0.03, sinkh="ot"):
    n = P.shape[0]
    power = -1 if maximize else 1
    lamb = reg / np.max(np.abs(P))

    if sinkh == "ot":
        ones = np.ones(n)
        P_eps = sinkhorn(
            ones, ones, P, power / lamb, stopInnerThr=tol, numItermax=500
        )  # * (P > np.log(1/n)/lamb)
    elif sinkh == "ds":
        lamb = reg / np.max(np.abs(P))
        P = np.exp(lamb * power * P)
        P_eps = _doubly_stochastic(P, tol, 500)

    #     return np.around(P_eps,3)
    #    r = int(np.floor(np.log10(1/n)) * -2)
    #    P_eps = np.around(P_eps, r)

    return P_eps


rows = []
for init in range(n_init):
    if verbose > 0:
        print(f"Initialization: {init}")
    shuffle_inds = np.random.permutation(n)
    correct_perm = np.argsort(shuffle_inds)
    A_base = ll_adj.copy()
    B_base = rr_adj.copy()
    AB_base = lr_adj.copy()
    BA_base = rl_adj.copy()

    for between_term in [False, True]:
        init_t0 = time.time()
        if verbose > 1:
            print(f"Between term: {between_term}")
        A = A_base
        B = B_base[shuffle_inds][:, shuffle_inds]
        AB = AB_for_obj = AB_base[:, shuffle_inds]
        BA = BA_for_obj = BA_base[shuffle_inds]

        if not between_term:
            AB = np.zeros((n, n))
            BA = np.zeros((n, n))

        P = P0.copy()
        P = P[:, shuffle_inds]

        if alpha > 0:
            rand_ds = np.random.uniform(size=(n, n))
            rand_ds = _doubly_stochastic(rand_ds)
            P = (1 - alpha) * P + alpha * rand_ds

        # _, iteration_perm = linear_sum_assignment(-P)
        # match_ratio = (correct_perm == iteration_perm)[:n_pairs].mean()
        # print(match_ratio)

        obj_func_scalar = 1
        if maximize:
            obj_func_scalar = -1

        for n_iter in range(1, maxiter + 1):

            # [1] Algorithm 1 Line 3 - compute the gradient of f(P)
            currtime = time.time()
            grad_fp = compute_gradient(A, B, AB, BA, P)
            if verbose > 2:
                print(f"{time.time() - currtime:.3f} seconds elapsed for grad_fp.")

            # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
            currtime = time.time()
            if ot:
                # TODO not implemented here yet
                Q = alap(grad_fp, reg, maximize=maximize, tol=thr)
            else:
                _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
                Q = np.eye(n)[cols]
            if verbose > 2:
                print(
                    f"{time.time() - currtime:.3f} seconds elapsed for LSAP/Sinkhorn step."
                )

            # [1] Algorithm 1 Line 5 - compute the step size
            currtime = time.time()

            alpha = compute_step_size(A, B, AB, BA, P, Q)

            if verbose > 2:
                print(
                    f"{time.time() - currtime:.3f} seconds elapsed for quadradic terms."
                )

            # [1] Algorithm 1 Line 6 - Update P
            P_i1 = alpha * P + (1 - alpha) * Q
            if np.linalg.norm(P - P_i1) / np.sqrt(n) < tol:
                P = P_i1
                break
            P = P_i1
            #
            #

            objfunc = compute_objective_function(A, B, AB_for_obj, BA_for_obj, P)

            _, perm = linear_sum_assignment(-P)

            # unshuffled_perm = np.zeros(n, dtype=int)
            # unshuffled_perm[perm_A] = perm_B[perm]

            match_ratio = (correct_perm == perm).mean()

            if verbose > 1:
                print(
                    f"Iter: {n_iter},  Objfunc: {objfunc:.2f}, Match ratio: {match_ratio:.2f}"
                )

            row = {
                "init": init,
                "iter": n_iter,
                "objfunc": objfunc,
                "between_term": between_term,
                "time": time.time() - init_t0,
                "P": P[:, correct_perm],
                "perm": perm,
                "match_ratio": match_ratio,
            }
            rows.append(row)

        if verbose > 1:
            print("\n")

    if verbose > 1:
        print("\n")

results = pd.DataFrame(rows)
results

#%%
last_results_idx = results.groupby(["between_term", "init"])["iter"].idxmax()
last_results = results.loc[last_results_idx].copy()

#%%
# TODO save the results

from giskard.plot import matched_stripplot

matched_stripplot(data=last_results, x="between_term", y="match_ratio", match="init")

# %%
from scipy.stats import wilcoxon

between_results = last_results[last_results["between_term"] == True]
no_between_results = last_results[last_results["between_term"] == False]

wilcoxon(
    between_results["match_ratio"].values, no_between_results["match_ratio"].values
)
