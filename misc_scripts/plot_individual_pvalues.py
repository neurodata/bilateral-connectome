#%% [markdown]
# # Comparing methods for SBM testing

#%%
from tkinter import N
from pkg.utils import set_warnings

set_warnings()

import csv
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import subuniformity_plot
from matplotlib.transforms import Bbox
from myst_nb import glue as default_glue
from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import binom_2samp, stochastic_block_test
from scipy.stats import beta, binom, chi2
from scipy.stats import combine_pvalues as scipy_combine_pvalues
from scipy.stats import ks_1samp, uniform
from tqdm import tqdm

DISPLAY_FIGS = True

FILENAME = "plot_individual_pvalues"


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


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()
fisher_color = sns.color_palette("Set2")[2]
min_color = sns.color_palette("Set2")[3]
eric_color = sns.color_palette("Set2")[4]
# method_palette = {"fisher": fisher_color, "min": min_color, "eric": eric_color}

GROUP_KEY = "simple_group"

left_adj, left_nodes = load_unmatched(side="left")
right_adj, right_nodes = load_unmatched(side="right")

left_labels = left_nodes[GROUP_KEY].values
right_labels = right_nodes[GROUP_KEY].values

#%%
stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="fisher",
    combine_method="fisher",
)

B_base = misc["probabilities1"].values
inds = np.nonzero(B_base)
base_probs = B_base[inds]
n_possible_matrix = misc["possible1"].values
ns = n_possible_matrix[inds]

#%%


def compare_individual_probabilities(counts1, n_possible1, counts2, n_possible2):
    pvalue_collection = []
    for i in range(len(counts1)):
        sub_stat, sub_pvalue = binom_2samp(
            counts1[i],
            n_possible1[i],
            counts2[i],
            n_possible2[i],
            null_odds=1,
            method="fisher",
        )
        pvalue_collection.append(sub_pvalue)

    pvalue_collection = np.array(pvalue_collection)
    # n_overall = len(pvalue_collection)
    # pvalue_collection = pvalue_collection[~np.isnan(pvalue_collection)]
    # n_tests = len(pvalue_collection)
    # n_skipped = n_overall - n_tests
    return pvalue_collection


save_path = Path(
    "/Users/bpedigo/JHU_code/bilateral/bilateral-connectome/results/"
    f"outputs/{FILENAME}/pvalues.csv"
)

n_sims = 200
n_perturb = 0
perturb_size = 0

all_pvalues = []

RERUN_SIM = False
if RERUN_SIM:
    for sim in tqdm(range(n_sims)):

        # choose some elements to perturb
        perturb_probs = base_probs.copy()
        choice_indices = rng.choice(len(perturb_probs), size=n_perturb, replace=False)

        # pertub em
        for index in choice_indices:
            prob = base_probs[index]

            new_prob = -1
            while new_prob <= 0 or new_prob >= 1:
                new_prob = rng.normal(prob, scale=prob * perturb_size)

            perturb_probs[index] = new_prob

        # sample some new binomial data
        base_samples = binom.rvs(ns, base_probs)
        perturb_samples = binom.rvs(ns, perturb_probs)

        pvalue_collection = compare_individual_probabilities(
            base_samples, ns, perturb_samples, ns
        )
        all_pvalues.append(pvalue_collection)

    all_pvalues = np.array(all_pvalues)

    np.savetxt(save_path, all_pvalues, delimiter=",")
else:
    all_pvalues = np.loadtxt(save_path, delimiter=",")

#%%

seed = 88888
np.random.seed(seed)
colors = sns.color_palette()
choice_inds = np.random.choice(len(ns), size=9, replace=False)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

sns.scatterplot(x=ns, y=base_probs, ax=ax, s=20, linewidth=0, alpha=0.7)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set(xlabel="n", ylabel="p")

for i, ind in enumerate(choice_inds):
    ax.scatter(ns[ind], base_probs[ind], color=colors[i + 1], s=40)


gluefig("n-prob-scatter", fig)

#%%
for i, ind in enumerate(choice_inds):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    uni_pvalues, counts = np.unique(all_pvalues[:, ind], return_counts=True)
    counts = counts / counts.sum()
    markerlines, stemlines, baseline = ax.stem(uni_pvalues, counts, markerfmt=".")
    plt.setp(stemlines, "color", colors[i + 1])
    plt.setp(markerlines, "color", colors[i + 1])
    plt.setp(markerlines, "markersize", 5)
    plt.setp(stemlines, "linewidth", 0.5)
    plt.setp(baseline, "color", "white")
    ylims = ax.get_ylim()
    ax.set_ylim((0, ylims[1]))
    ax.set(xlabel="pvalue", ylabel="Frequency")
    ax.text(
        0.05,
        0.95,
        f"n={ns[ind]}\np={base_probs[ind]:0.2g}\nnp={ns[ind]*base_probs[ind]:0.2g}",
        va="top",
        transform=ax.transAxes,
    )
    gluefig(f"pvalue-dist-example{i}", fig)
