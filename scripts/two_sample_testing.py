#%%

import numpy as np
import pandas as pd
import seaborn as sns
from pkg.plot import set_theme
from pkg.io import savefig
from myst_nb import glue as default_glue
import matplotlib.pyplot as plt

DISPLAY_FIGS = False

FILENAME = "two_sample_testing"


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


set_theme(font_scale=1.5)

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
colors = sns.color_palette()
pad = 0.3
sns.stripplot(data=data, x="labels", y="y", ax=ax, s=10, jitter=pad)
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

gluefig("2_sample_real_line", fig)
