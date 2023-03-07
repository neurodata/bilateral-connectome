#%%
import datetime
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pkg.data import load_network_palette, load_node_palette, load_unmatched
from pkg.io import FIG_PATH, get_environment_variables
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.stats import binom_2samp, stochastic_block_test
from scipy.stats import binom

from tqdm.autonotebook import tqdm


_, _, DISPLAY_FIGS = get_environment_variables()

FILENAME = "sbm_block_power"

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)

network_palette, NETWORK_KEY = load_network_palette()
node_palette, NODE_KEY = load_node_palette()
neutral_color = sns.color_palette("Set2")[2]

GROUP_KEY = "celltype_discrete"

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
)
glue("pvalue", pvalue, form="pvalue")
n_tests = misc["n_tests"]
glue("n_tests", n_tests)

#%%

possible1 = misc["possible1"]
possible2 = misc["possible2"]
probs1 = misc["probabilities1"]
probs2 = misc["probabilities2"]


# %%


method = "fisher"
index = possible1.index
n_sims = 100
effect_scale = 0.8
count = 0
rows = []
pbar = tqdm(total=n_sims * len(index) ** 2)
for source_group in index:
    for target_group in index:
        p1 = probs1.loc[source_group, target_group]
        p2 = probs2.loc[source_group, target_group]
        mean_p = (p1 + p2) / 2
        n1 = possible1.loc[source_group, target_group]
        n2 = possible2.loc[source_group, target_group]

        for sim in range(n_sims):
            edges1 = binom.rvs(n1, mean_p, random_state=rng)
            edges2 = binom.rvs(n2, effect_scale * mean_p, random_state=rng)
            stat, pvalue = binom_2samp(edges1, n1, edges2, n2, method=method)
            rows.append(
                {
                    "source_group": source_group,
                    "target_group": target_group,
                    "n1": n1,
                    "n2": n2,
                    "mean_p": mean_p,
                    "sim": sim,
                    "stat": stat,
                    "pvalue": pvalue,
                }
            )
            pbar.update(1)
pbar.close()

results = pd.DataFrame(rows)
#%%


def compute_power(pvalues, alpha=0.05):
    return np.mean(pvalues < alpha)


power_results = (
    results.groupby(["source_group", "target_group"])["pvalue"]
    .apply(compute_power)
    .rename("power")
    .reset_index()
)

power_results

#%%
square_power = power_results.pivot(
    index="source_group", columns="target_group", values="power"
)
square_power

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(
    square_power,
    square=True,
    cmap="RdBu_r",
    vmin=0,
    center=0,
    vmax=1,
    cbar_kws=dict(shrink=0.5, pad=0.1),
    ax=ax,
)
ax.set(ylabel="Source group", xlabel="Target group")
cax = fig.get_axes()[1]
cax.set_title("Power @\n" + r"$\alpha=0.05$", pad=20)
gluefig("empirical_power_by_block", fig)

#%%
mean_possible = (possible1 + possible2) / 2

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(
    mean_possible,
    square=True,
    cmap="RdBu_r",
    vmin=0,
    center=0,
    cbar_kws=dict(shrink=0.5, pad=0.1),
    ax=ax,
)
ax.set(ylabel="Source group", xlabel="Target group")
cax = fig.get_axes()[1]
cax.set_title("# possible\nedges", pad=10)
gluefig("n_possible_by_block", fig)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
