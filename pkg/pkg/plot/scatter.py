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


def scattermap(data, ax=None, legend=False, sizes=(5, 10), hue=None, **kws):
    r"""
    Draw a matrix using points instead of a heatmap. Helpful for larger, sparse
    matrices.
    Parameters
    ----------
    data : np.narray, scipy.sparse.csr_matrix, ndim=2
        Matrix to plot
    ax: matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    legend : bool, optional
        [description], by default False
    sizes : tuple, optional
        min and max of dot sizes, by default (5, 10)
    spines : bool, optional
        whether to keep the spines of the plot, by default False
    border : bool, optional
        [description], by default True
    **kws : dict, optional
        Additional plotting arguments
    Returns
    -------
    ax: matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
    n_verts = data.shape[0]
    inds = np.nonzero(data)
    edges = np.squeeze(np.asarray(data[inds]))
    scatter_df = pd.DataFrame()
    scatter_df["weight"] = edges
    scatter_df["x"] = inds[1]
    scatter_df["y"] = inds[0]
    if hue is not None:
        scatter_df["hue"] = hue
    sns.scatterplot(
        data=scatter_df,
        x="x",
        y="y",
        size="weight",
        legend=legend,
        sizes=sizes,
        ax=ax,
        linewidth=0,
        hue=hue,
        **kws,
    )
    ax.set_xlim((-0.5, n_verts - 0.5))
    ax.set_ylim((n_verts - 0.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    return ax
