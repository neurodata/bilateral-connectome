import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint


def plot_density(misc, palette=None, ax=None, coverage=0.95):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))

    n_possible_left = misc["possible1"]
    n_possible_right = misc["possible2"]

    density_left = misc["probability1"]
    density_right = misc["probability2"]

    n_edges_left = misc["observed1"]
    n_edges_right = misc["observed2"]

    # ax.bar(0, density_left, color=palette["Left"])
    # ax.bar(1, density_right, color=palette["Right"])

    left_lower, left_upper = proportion_confint(
        n_edges_left, n_possible_left, alpha=1 - coverage, method="beta"
    )
    right_lower, right_upper = proportion_confint(
        n_edges_right, n_possible_right, alpha=1 - coverage, method="beta"
    )

    halfwidth = 0.1
    linewidth = 4

    color = palette["Left"]
    x = 0
    ax.plot(
        [x - halfwidth, x + halfwidth],
        [density_left, density_left],
        color=color,
        linewidth=linewidth,
    )
    ax.plot([x, x], [left_lower, left_upper], color=color, linewidth=linewidth)

    color = palette["Right"]
    x = 1
    ax.plot(
        [x - halfwidth, x + halfwidth],
        [density_right, density_right],
        color=color,
        linewidth=linewidth,
    )
    ax.plot([x, x], [right_lower, right_upper], color=color, linewidth=linewidth)

    yticks = [np.round(density_left, 4), np.round(density_right, 4)]

    ax.set(
        xlabel="Hemisphere",
        xticks=[0, 1],
        xticklabels=["Left", "Right"],
        xlim=(-0.5, 1.5),
        yticks=yticks,
        # ylim=(0, max(right_upper, left_upper) * 1.05),
        ylabel=r"Estimated density ($\hat{p}$)",
    )

    labels = ax.get_xticklabels()
    for label in labels:
        text_string = label.get_text()
        label.set_color(palette[text_string])

    return ax.get_figure(), ax
