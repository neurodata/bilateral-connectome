import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox


def shrink_axis(ax, scale=0.7):
    pos = ax.get_position()
    mid = (pos.ymax + pos.ymin) / 2
    height = pos.ymax - pos.ymin
    new_pos = Bbox(
        [
            [pos.xmin, mid - scale * 0.5 * height],
            [pos.xmax, mid + scale * 0.5 * height],
        ]
    )
    ax.set_position(new_pos)


def draw_colors(ax, ax_type="x", labels=None, palette="tab10"):
    r"""
    Draw colormap onto the axis to separate the data
    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the colormap
    ax_type : char, optional
        Setting either the x or y axis, by default "x"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    divider : AxesLocator, optional
        Divider used to add new axes to the plot
    color : str, list of str, or array_like, optional
        Attribute in meta by which to draw colorbars, by default None
    palette : str or dict, optional
        Colormap of the colorbar, by default "tab10"
    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the color map
    """

    uni_classes = np.unique(labels)
    # Create the color dictionary
    color_dict = palette

    # Make the colormap
    class_map = dict(zip(uni_classes, range(len(uni_classes))))
    color_sorted = np.vectorize(color_dict.get)(uni_classes)
    color_sorted = np.array(color_sorted).T
    lc = ListedColormap(color_sorted)
    class_indicator = np.vectorize(class_map.get)(labels)

    if ax_type == "x":
        class_indicator = class_indicator.reshape(1, len(labels))
    elif ax_type == "y":
        class_indicator = class_indicator.reshape(len(labels), 1)
    sns.heatmap(
        class_indicator,
        cmap=lc,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        ax=ax,
        square=False,
    )
    return ax


def remove_shared_ax(ax):
    """
    Remove ax from its sharex and sharey
    """
    # Remove ax from the Grouper object
    shax = ax.get_shared_x_axes()
    shay = ax.get_shared_y_axes()
    shax.remove(ax)
    shay.remove(ax)

    # Set a new ticker with the respective new locator and formatter
    for axis in [ax.xaxis, ax.yaxis]:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        # No ticks and no labels
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)
