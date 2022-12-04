#%%
from pkg.data import load_matched

left_adj, left_nodes = load_matched("left", weights=False)
right_adj, right_nodes = load_matched("right", weights=False)
assert (left_nodes["pair_id"].values == right_nodes["pair_id"].values).all()

#%%
max_n_components = 48

from graspologic.embed import AdjacencySpectralEmbed


ase = AdjacencySpectralEmbed(n_components=max_n_components)
X_left, Y_left = ase.fit_transform(left_adj)
X_right, Y_right = ase.fit_transform(right_adj)

#%%

import numpy as np


def regularize_P(P):
    P -= np.diag(np.diag(P))
    n = P.shape[0]
    lower_lim = 1 / n**2
    upper_lim = 1 - lower_lim
    P[P >= upper_lim] = upper_lim
    P[P <= lower_lim] = lower_lim
    return P


def compute_P(X, Y):
    P = X @ Y.T
    P = regularize_P(P)
    return P


#%%
from scipy.stats import bernoulli


#%%
from tqdm.notebook import tqdm

n_components_range = np.arange(1, max_n_components + 1)
rows = []
for n_components in tqdm(n_components_range):
    P_left = compute_P(X_left[:, :n_components], Y_left[:, :n_components])
    log_lik = bernoulli(P_left).logpmf(right_adj).sum()
    norm_log_lik = log_lik / right_adj.sum()
    rows.append(
        {
            "n_components": n_components,
            "log_lik": log_lik,
            "norm_log_lik": norm_log_lik,
            "train_side": "left",
            "test_side": "right",
        }
    )
for n_components in tqdm(n_components_range):
    P_right = compute_P(X_right[:, :n_components], Y_right[:, :n_components])
    log_lik = bernoulli(P_right).logpmf(left_adj).sum()
    norm_log_lik = log_lik / left_adj.sum()
    rows.append(
        {
            "n_components": n_components,
            "log_lik": log_lik,
            "norm_log_lik": norm_log_lik,
            "train_side": "right",
            "test_side": "left",
        }
    )

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

results = pd.DataFrame(rows)
y = "norm_log_lik"
sns.lineplot(data=results, x="n_components", y=y, hue="train_side", ax=ax)

for side, side_results in results.groupby("train_side"):
    idx = side_results[y].idxmax()
    n_components = side_results.loc[idx, "n_components"]
    ax.axvline(n_components, color="k", linestyle="--", alpha=0.5)
    print(n_components)

#%%


from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.utils import binarize, remove_loops
from scipy.stats import poisson


def calculate_blockmodel_likelihood(source_adj, target_adj, labels, model="DCSBM"):
    if model == "DCSBM":
        Model = DCSBMEstimator
    elif model == "SBM":
        Model = SBMEstimator

    estimator = Model(directed=True, loops=False)
    uni_labels, inv = np.unique(labels, return_inverse=True)
    estimator.fit(source_adj, inv)
    train_P = estimator.p_mat_
    train_P = regularize_P(train_P)

    n_params = estimator._n_parameters() + len(labels)

    score = poisson.logpmf(target_adj, train_P).sum()
    out = dict(
        score=score,
        model=model,
        n_params=n_params,
        norm_score=score / target_adj.sum(),
        n_communities=len(uni_labels),
    )

    return out


#%%

n_components = 24
method = "average"
metric = "cosine"
from scipy.cluster import hierarchy

n_clusters_range = np.arange(5, 125, 1)

rows = []
for (X_fit, adj_fit), side_fit in zip(
    [(X_left, left_adj), (X_right, right_adj)], ["left", "right"]
):
    Z = hierarchy.linkage(X_fit[:, :n_components], method=method, metric=metric)
    for n_clusters in tqdm(n_clusters_range):
        labels = hierarchy.fcluster(Z, n_clusters, criterion="maxclust")
        for adj_eval, side_eval in zip([left_adj, right_adj], ["left", "right"]):
            row = calculate_blockmodel_likelihood(adj_fit, adj_eval, labels)
            row["side_fit"] = side_fit
            row["side_eval"] = side_eval
            rows.append(row)

results = pd.DataFrame(rows)
results
# %%
results["fit_eval"] = results["side_fit"] + "-" + results["side_eval"]
results["is_same"] = results["side_fit"] == results["side_eval"]
from pkg.plot import set_theme

set_theme()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(
    data=results.query("model == 'DCSBM'"),
    x="n_communities",
    y="norm_score",
    hue="side_eval",
    style="is_same",
)

# %%
mean_results = (
    results.groupby(["n_communities", "is_same"]).mean(numeric_only=True).reset_index()
)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(
    data=mean_results,
    x="n_communities",
    y="norm_score",
    style="is_same",
)


#%%

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from graspologic.utils import pass_to_ranks, symmetrize
from scipy.stats.mstats import gmean


def preprocess_nblast(nblast_scores, symmetrize_mode="geom", transform="ptr"):
    if isinstance(nblast_scores, np.ndarray):
        distance = nblast_scores  # the raw nblast scores are dissimilarities/distances
        index = None
    else:
        distance = nblast_scores.values
        index = nblast_scores.index
    indices = np.triu_indices_from(distance, k=1)

    if symmetrize_mode == "geom":
        fwd_dist = distance[indices]
        back_dist = distance[indices[::-1]]
        stack_dist = np.concatenate(
            (fwd_dist.reshape(-1, 1), back_dist.reshape(-1, 1)), axis=1
        )
        geom_mean = gmean(stack_dist, axis=1)
        sym_distance = np.zeros_like(distance)
        sym_distance[indices] = geom_mean
        sym_distance[indices[::-1]] = geom_mean
    else:  # simple average
        sym_distance = symmetrize(distance)
    # make the distances between 0 and 1
    sym_distance /= sym_distance.max()
    sym_distance -= sym_distance.min()
    # and then convert to similarity
    morph_sim = 1 - sym_distance

    if transform == "quantile":
        quant = QuantileTransformer(n_quantiles=2000)
        transformed_vals = quant.fit_transform(morph_sim[indices].reshape(-1, 1))
        transformed_vals = np.squeeze(transformed_vals)
        transformed_morph = np.ones_like(morph_sim)
        transformed_morph[indices] = transformed_vals
        transformed_morph[indices[::-1]] = transformed_vals
    elif transform == "ptr":
        transformed_morph = pass_to_ranks(morph_sim)
        np.fill_diagonal(
            transformed_morph, 1
        )  # should be exactly 1, isnt cause of ties
    elif transform == "log":
        transformed_morph = np.log(morph_sim) + 1
    else:
        transformed_morph = morph_sim

    if index is not None:
        transformed_morph = pd.DataFrame(transformed_morph, index=index, columns=index)

    return transformed_morph


symmetrize_mode = "geom"
transform = "ptr"
nblast_type = "scores"

from pathlib import Path

data_dir = Path(
    "/Users/bpedigo/JHU_code/maggot_models/maggot_models/experiments/nblast/outs"
)


side = "left"
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)

# intersect_index = np.intersect1d(nblast_sim.index, left_nodes.index)
left_sim = preprocess_nblast(
    nblast_sim, symmetrize_mode=symmetrize_mode, transform=transform
)

nblast_sim = left_sim.reindex(index=left_nodes.index, columns=left_nodes.index)
valid = ~nblast_sim.isna().all(axis=1)

#%%
cluster_win_nblasts = []
valid_labels = labels[valid]
uni_clusters = np.unique(valid_labels)
for label in uni_clusters:
    within_sim = nblast_sim.values[labels == label][:, labels == label]
    triu_inds = np.triu_indices_from(within_sim, k=1)
    upper = within_sim[triu_inds]
    if len(upper) > 0:
        mean_within_sim = np.nanmean(upper)
    else:
        mean_within_sim = np.nan
    cluster_win_nblasts.append(mean_within_sim)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(x=cluster_win_nblasts, ax=ax)
