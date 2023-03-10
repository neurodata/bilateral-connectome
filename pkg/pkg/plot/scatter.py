import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import legend_upper_right


def scatterplot(
    data,
    x=None,
    y=None,
    hue=None,
    shift=None,
    shift_bounds=(-0.1, 0.1),
    ax=None,
    shade=False,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
    data = data.copy()
    data["x_shift"] = data[x]
    if shift is not None:
        groups = data.groupby(shift, sort=False)
        shifts = np.linspace(shift_bounds[0], shift_bounds[1], len(groups))
        shifts = dict(zip(groups.groups.keys(), shifts))
        for group_key, group_data in groups:
            data.loc[group_data.index, "x_shift"] += shifts[group_key]
    sns.scatterplot(data=data, x="x_shift", y=y, hue=hue, ax=ax, **kwargs)
    ax.set_xlabel(x)
    start = int(data[x].unique().min())
    stop = int(data[x].unique().max())
    if shade > 0:
        # xlim = ax.get_xlim()
        for x in np.arange(start, stop + 1, 2):
            ax.axvspan(x - 0.5, x + 0.5, color="lightgrey", alpha=0.2, linewidth=0)
    legend_upper_right(ax)
    return ax
