import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from graspologic.plot import heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn.utils import relative_luminance
from pathlib import Path

from .theme import set_theme
from .utils import bound_texts, draw_colors, remove_shared_ax, shrink_axis


def plot_stochastic_block_probabilities(misc, network_palette):
    # get values
    B1 = misc["probabilities1"]
    B2 = misc["probabilities2"]
    null_odds = misc["null_ratio"]
    B2 = B2 * null_odds

    p_max = max(B1.values.max(), B2.values.max())
    K = B1.shape[0]

    # set up plot
    pad = 2
    width_ratios = [0.5, pad + 0.8, 10, pad, 10]
    set_theme(font_scale=1.25)
    fig, axs = plt.subplots(
        1,
        len(width_ratios),
        figsize=(20, 10),
        gridspec_kw=dict(
            width_ratios=width_ratios,
        ),
    )
    left_col = 2
    right_col = 4
    # pvalue_col = 6

    heatmap_kws = dict(
        cmap="Blues", square=True, cbar=False, vmax=p_max, fmt="s", xticklabels=True
    )

    # heatmap of left connection probabilities
    annot = np.full((K, K), "")
    annot[B1.values == 0] = 0
    ax = axs[left_col]
    sns.heatmap(B1, ax=ax, annot=annot, **heatmap_kws)
    ax.set(ylabel="Source group", xlabel="Target group")
    ax.set_title(r"$\hat{B}$ left", fontsize="xx-large", color=network_palette["Left"])

    # heatmap of right connection probabilities
    annot = np.full((K, K), "")
    annot[B2.values == 0] = 0
    ax = axs[right_col]
    im = sns.heatmap(B2, ax=ax, annot=annot, **heatmap_kws)
    ax.set(ylabel="", xlabel="Target group")
    text = r"$\hat{B}$ right"
    if null_odds != 1:
        text = r"$c$" + text
    ax.set_title(text, fontsize="xx-large", color=network_palette["Right"])
    # ax.set(yticks=[], yticklabels=[])

    # handle the colorbars
    # NOTE: did it this way cause the other options weren't playing nice with auto
    # constrain
    # layouts.

    ax = axs[0]
    shrink_axis(ax, scale=0.5)
    _ = fig.colorbar(
        im.get_children()[0],
        cax=ax,
        fraction=1,
        shrink=1,
        ticklocation="left",
    )
    ax.set_title("Estimated\nedge\nprobability")

    # remove dummy axes
    for i in range(len(width_ratios)):
        if not axs[i].has_data():
            axs[i].set_visible(False)

    ax = axs[left_col]

    texts = []
    texts.append(ax.text(-0.5, -0.16, "0 - No edges", transform=ax.transAxes))
    texts.append(ax.text(-0.41, -0.22, "observed", transform=ax.transAxes))

    bound_texts(
        texts, ax=ax, facecolor="white", edgecolor="lightgrey", xpad=0.3, ypad=1.2
    )

    return fig, axs


def plot_pvalues(
    misc,
    pvalue_vmin=None,
    ax=None,
    cax=None,
    annot_missing=True,
):
    if pvalue_vmin is None:
        import json

        vars_file = Path(__file__).parent.parent.parent.parent
        vars_file = vars_file / "results"
        vars_file = vars_file / "glued_variables.json"

        with open(vars_file, "r") as f:
            vars_dict = json.load(f)
            pvalue_vmin = vars_dict["sbm_unmatched_test-pvalue_vmin"]

    if ax is None:
        width_ratios = [0.5, 3, 10]
        fig, axs = plt.subplots(
            1,
            3,
            figsize=(10, 10),
            gridspec_kw=dict(
                width_ratios=width_ratios,
            ),
        )
        axs[1].remove()
        ax = axs[-1]
        cax = axs[0]

    corrected_pvalues = misc["corrected_pvalues"]
    B1 = misc["probabilities1"]
    B2 = misc["probabilities2"]

    K = len(B1)
    index = B1.index
    if annot_missing:
        annot = np.full((K, K), "")
        annot[(B1.values == 0) & (B2.values == 0)] = "B"
        annot[(B1.values == 0) & (B2.values != 0)] = "L"
        annot[(B1.values != 0) & (B2.values == 0)] = "R"
    else:
        annot = False
    plot_pvalues = np.log10(corrected_pvalues)
    plot_pvalues[np.isnan(plot_pvalues)] = 0
    im = sns.heatmap(
        plot_pvalues,
        ax=ax,
        annot=annot,
        cmap="RdBu",
        center=0,
        square=True,
        cbar=False,
        fmt="s",
        vmin=pvalue_vmin,
    )
    ax.set(ylabel="Source group", xlabel="Target group")
    ax.set(xticks=np.arange(K) + 0.5, xticklabels=index)
    # ax.set_title(r"Probability comparison", fontsize="x-large")

    colors = im.get_children()[0].get_facecolors()
    significant = misc["rejections"]

    # NOTE: the x's looked bad so I did this super hacky thing...
    pad = 0.2
    for idx, (is_significant, color) in enumerate(
        zip(significant.values.ravel(), colors)
    ):
        if is_significant:
            i, j = np.unravel_index(idx, (K, K))
            # REF: seaborn heatmap
            lum = relative_luminance(color)
            text_color = ".15" if lum > 0.408 else "w"

            xs = [j + pad, j + 1 - pad]
            ys = [i + pad, i + 1 - pad]
            ax.plot(xs, ys, color=text_color, linewidth=4)
            xs = [j + 1 - pad, j + pad]
            ys = [i + pad, i + 1 - pad]
            ax.plot(xs, ys, color=text_color, linewidth=4)

    # plot colorbar for the pvalue plot
    # NOTE: only did it this way for consistency with the other colorbar
    shrink_axis(cax, scale=0.5, shift=0.05)
    fig = ax.get_figure()
    _ = fig.colorbar(
        im.get_children()[0],
        cax=cax,
        fraction=1,
        shrink=1,
        ticklocation="left",
    )
    cax.set_title(r"$log_{10}$" + "\ncorrected" "\np-value", pad=20)

    cax.plot(
        [0, 1], [np.log10(0.05), np.log10(0.05)], zorder=100, color="black", linewidth=3
    )
    cax.annotate(
        r"$\alpha$",
        (0.05, np.log10(0.05)),
        xytext=(-20, 15),
        textcoords="offset points",
        va="center",
        ha="right",
        arrowprops={"arrowstyle": "-", "linewidth": 3, "relpos": (0, 0.5)},
    )

    if annot_missing:
        texts = []
        texts.append(ax.text(-0.60, -0.1, "Test not run,", transform=ax.transAxes))
        texts.append(ax.text(-0.60, -0.16, "no edges on:", transform=ax.transAxes))
        texts.append(ax.text(-0.55, -0.24, "L - left", transform=ax.transAxes))
        texts.append(ax.text(-0.55, -0.3, "R - right", transform=ax.transAxes))
        texts.append(ax.text(-0.55, -0.36, "B - both", transform=ax.transAxes))

        bound_texts(
            texts, ax=ax, facecolor="white", edgecolor="lightgrey", xpad=0.3, ypad=1.2
        )

    texts = []
    texts.append(ax.text(-0.21, -0.36, "X - significant", transform=ax.transAxes))
    bound_texts(
        texts, ax=ax, facecolor="white", edgecolor="lightgrey", xpad=0.3, ypad=1.2
    )

    return fig, axs


def heatmap_grouped(
    Bhat,
    labels,
    palette=None,
    ax=None,
    pad=0,
    color_size="5%",
    vmin=0,
    vmax=1,
    cmap="Blues",
    center=None,
    title="",
    xlabel="",
    ylabel="",
    xlabel_loc="top",
):
    heatmap(Bhat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=center, cbar=False)
    divider = make_axes_locatable(ax)
    top_ax = divider.append_axes(xlabel_loc, size=color_size, pad=pad, sharex=ax)
    remove_shared_ax(top_ax)
    draw_colors(top_ax, "x", labels=labels, palette=palette)
    left_ax = divider.append_axes("left", size=color_size, pad=pad, sharex=ax)
    remove_shared_ax(left_ax)
    draw_colors(left_ax, "y", labels=labels, palette=palette)
    if xlabel_loc == "top":
        top_ax.set_title(title)
        ax.set_xlabel(xlabel)
    else:
        ax.set_title(title)
        top_ax.set_xlabel(xlabel)
    left_ax.set_ylabel(ylabel)
    return top_ax, left_ax


def compare_probability_row(
    source, target, Bhat1, Bhat2, y, cmap=None, palette=None, ax=None
):
    x1 = 0.1
    x2 = 0.25
    sns.scatterplot(
        x=[x1, x2],
        y=[y, y],
        hue=[source, target],
        linewidth=1,
        edgecolor="black",
        palette=palette,
        ax=ax,
        legend=False,
        s=100,
    )
    # ax.arrow(x1, y, x2 - x1, 0, arrowprops=dict(arrowstyle='->'))
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="simple",
            connectionstyle="arc3,rad=-0.7",
            facecolor="black",
            shrinkA=5,
            shrinkB=5,
            # mutation_scale=,
        ),
    )

    x3 = 0.4

    size = 0.1
    phat = Bhat1[source - 1, target - 1]
    color = cmap(phat)
    patch = mpl.patches.Rectangle(
        (x3, y - size / 4), width=size, height=size / 2, facecolor=color
    )
    ax.add_patch(patch)

    text = ax.text(0.645, y, "?", ha="center", va="center")
    text.set_bbox(dict(facecolor="white", edgecolor="white"))

    x4 = 0.8
    phat = Bhat2[source - 1, target - 1]
    color = cmap(phat)
    patch = mpl.patches.Rectangle(
        (x4, y - size / 4), width=size, height=size / 2, facecolor=color
    )
    ax.add_patch(patch)

    ax.plot([x3, x4], [y, y], linewidth=2.5, linestyle=":", color="grey", zorder=-1)
