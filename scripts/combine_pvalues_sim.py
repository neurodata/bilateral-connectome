#%% [markdown]
# # Fisher's method vs. min (after multiple comparison's correction)

#%%
from pkg.utils import set_warnings

set_warnings()

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
from scipy.stats import binom, combine_pvalues, ks_1samp, uniform
from tqdm import tqdm


DISPLAY_FIGS = False

FILENAME = "combine_pvalues_sim"


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

#%%
uncorrected_pvalue_path = Path(
    "/Users/bpedigo/JHU_code/bilateral/bilateral-connectome/results/"
    "outputs/compare_sbm_methods_sim/uncorrected_pvalues.csv"
)

uncorrected_pvalues_df = pd.read_csv(uncorrected_pvalue_path)

#%%
from ast import literal_eval
from scipy.stats import combine_pvalues

methods = ["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]
scipy_methods = ["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]

rows = []

for i, row in uncorrected_pvalues_df.iterrows():
    uncorrected_pvalues = np.array(literal_eval(row["uncorrected_pvalues"]))
    for method in methods:
        if method in scipy_methods:
            uncorrected_pvalues[uncorrected_pvalues == 1.0] = 0.99999
            # uncorrected_pvalues[uncorrected_pvalues == 0]
            if uncorrected_pvalues.min() == 0:
                stat = np.nan
                overall_pvalue = 0
            else:
                stat, overall_pvalue = combine_pvalues(
                    uncorrected_pvalues, method=method
                )
        new_row = row.copy()
        new_row["stat"] = stat
        new_row["pvalue"] = overall_pvalue
        new_row["method"] = method
        rows.append(new_row)

results = pd.DataFrame(rows, index=range(len(rows)))
results

#%%
lower_limit = 1e-10
lowers = results[results["pvalue"] < lower_limit].index
results.loc[lowers, "pvalue"] = lower_limit

#%%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

perturb_size_range = np.unique(results["perturb_size"])
for i, perturb_size in enumerate(perturb_size_range):
    ax = axs.flat[i]
    plot_results = results[results["perturb_size"] == perturb_size]
    sns.lineplot(
        data=plot_results,
        x="n_perturb",
        y="pvalue",
        hue="method",
        style="method",
        ax=ax,
    )
    ax.set(yscale="log")
    ax.get_legend().remove()
    ax.axhline(0.05, color="dimgrey", linestyle=":")
    ax.axhline(0.005, color="dimgrey", linestyle="--")
    ax.set(ylabel="", xlabel="", title=f"{perturb_size}")

    ylim = ax.get_ylim()
    if ylim[0] < lower_limit:
        ax.set_ylim((lower_limit / 10, ylim[1]))

handles, labels = ax.get_legend_handles_labels()

ax.annotate(
    0.05,
    xy=(ax.get_xlim()[1], 0.05),
    xytext=(30, 10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-"),
)
ax.annotate(
    0.005,
    xy=(ax.get_xlim()[1], 0.005),
    xytext=(30, -40),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-"),
)
axs.flat[-1].axis("off")

[ax.set(ylabel="p-value") for ax in axs[:, 0]]
[ax.set(xlabel="Number perturbed") for ax in axs[1, :]]
axs[0, -1].set(xlabel="Number perturbed")

axs[0, 0].set_title(f"Perturbation size = {perturb_size_range[0]}")

for i, label in enumerate(labels):
    labels[i] = label.capitalize()
axs.flat[-1].legend(handles=handles, labels=labels, title="Method")

# gluefig("perturbation_pvalues_lineplots", fig)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(
    x=plot_results["n_perturb"],
    y=plot_results["pvalue"],
    ax=ax
    # hue="method",
    # style="method",
)

#%%
uncorrected_pvalues = np.array(
    literal_eval(uncorrected_pvalues_df.iloc[0]["uncorrected_pvalues"])
)
sns.histplot(x=uncorrected_pvalues)
print(uncorrected_pvalues.min())
uncorrected_pvalues[uncorrected_pvalues == 1.0] = 0.99999
print(uncorrected_pvalues.max())
combine_pvalues(uncorrected_pvalues, method="pearson")

# %%
