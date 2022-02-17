import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


def plot_density(misc, palette=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    n_possible_left = misc["possible1"]
    n_possible_right = misc["possible2"]

    density_left = misc["probability1"]
    density_right = misc["probability2"]

    n_edges_left = misc["observed1"]
    n_edges_right = misc["observed2"]

    ax.bar(0, density_left, color=palette["Left"])
    ax.bar(1, density_right, color=palette["Right"])

    coverage = 0.99

    left_lower, left_upper = proportion_confint(
        n_edges_left, n_possible_left, alpha=1 - coverage, method="beta"
    )
    right_lower, right_upper = proportion_confint(
        n_edges_right, n_possible_right, alpha=1 - coverage, method="beta"
    )

    ax.plot([0, 0], [left_lower, left_upper], color="black", linewidth=4)
    ax.plot([1, 1], [right_lower, right_upper], color="black", linewidth=4)

    ax.set(
        xlabel="Hemisphere",
        xticks=[0, 1],
        xticklabels=["Left", "Right"],
        ylabel=r"Estimated density ($\hat{p}$)",
    )
    return ax
