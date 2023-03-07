#%%

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import load_network_palette
from pkg.io import FIG_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme

DISPLAY_FIGS = True

FILENAME = "two_sample_testing"

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
rng = np.random.default_rng(8888)

set_theme(font_scale=1.5)

palette, _ = load_network_palette()

colors = [palette["Left"], palette["Right"]]
palette = dict(zip([0, 1], colors))
rng = np.random.default_rng(8888)

x1 = rng.normal(1, 1, size=20)
x1_mean = np.mean(x1)
x2 = rng.normal(2, 1, size=20)
x2_mean = np.mean(x2)
data = np.concatenate((x1, x2))
labels = 20 * [0] + 20 * [1]

data = pd.DataFrame(data, columns=["y"])
data["labels"] = labels

fig, ax = plt.subplots(1, 1, figsize=(6, 8))

np.random.seed(8888)
pad = 0.3
sns.stripplot(data=data, x="labels", y="y", ax=ax, s=10, jitter=pad, palette=palette)
ax.set(yticks=[], xticklabels=["Group 1", "Group 2"], xlabel="")
ax.plot([-pad, pad], [x1_mean, x1_mean], color=colors[0], linewidth=3)
ax.text(
    pad * 1.1,
    x1_mean,
    r"$\bar{y}_1$",
    color=colors[0],
    ha="left",
    va="center",
    fontsize="medium",
)
ax.plot([-pad + 1, pad + 1], [x2_mean, x2_mean], color=colors[1], linewidth=3)
ax.text(
    1 + pad * 1.1,
    x2_mean,
    r"$\bar{y}_2$",
    color=colors[1],
    ha="left",
    va="center",
    fontsize="medium",
)
ax.spines.bottom.set_visible(False)
for i, text in enumerate(ax.xaxis.get_ticklabels()):
    text.set_color(colors[i])

fig.set_facecolor("w")

gluefig("2_sample_real_line", fig)

# %%

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

set_theme(font_scale=2)
sns.stripplot(data=data, x="labels", y="y", ax=ax, s=10, jitter=pad, palette=palette)
ax.set(yticks=[], xticklabels=["Group 1", "Group 2"], xlabel="")
# ax.plot([-pad, pad], [x1_mean, x1_mean], color=colors[0], linewidth=3)
# ax.text(
#     pad * 1.1,
#     x1_mean,
#     r"$\bar{y}_1$",
#     color=colors[0],
#     ha="left",
#     va="center",
#     fontsize="large",
# )
# ax.plot([-pad + 1, pad + 1], [x2_mean, x2_mean], color=colors[1], linewidth=3)
# ax.text(
#     1 + pad * 1.1,
#     x2_mean,
#     r"$\bar{y}_2$",
#     color=colors[1],
#     ha="left",
#     va="center",
#     fontsize="large",
# )
ax.spines.bottom.set_visible(False)
for i, text in enumerate(ax.xaxis.get_ticklabels()):
    text.set_color(colors[i])

fig.set_facecolor("w")

gluefig("2_sample_real_line_wide", fig)

