import numpy as np
import pandas as pd
import pymaid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import navis

volume_names = ["PS_Neuropil_manual"]


def rgb2hex(r, g, b):
    r = int(255 * r)
    g = int(255 * g)
    b = int(255 * b)

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def simple_plot_neurons(
    neurons,
    azim=-90,
    elev=-90,
    dist=5,
    use_x=True,
    use_y=True,
    use_z=False,
    palette=None,
    volume_names=volume_names,
    ax=None,
    autoscale=False,
    axes_equal=True,
    force_bounds=True,
    axis_off=True,
    **kwargs,
):
    if isinstance(neurons, (list, np.ndarray, pd.Series, pd.Index)):
        try:
            neuron_ids = [int(n.id) for n in neurons]
        except AttributeError:
            neuron_ids = neurons
        neurons = []
        for n in neuron_ids:
            try:
                neuron = pymaid.get_neuron(n)
                neurons.append(neuron)
            except:
                print(f"Error when retreiving neuron skeleton {n}")
    elif isinstance(neurons, navis.NeuronList):
        neuron_ids = neurons.id
        neuron_ids = [int(n) for n in neuron_ids]

    for key, value in palette.items():
        if isinstance(value, tuple):
            palette[key] = rgb2hex(*value)

    # neurons = [pymaid.get_neuron(n) for n in neuron_ids]
    # volumes = [pymaid.get_volume(v) for v in volume_names]
    colors = np.vectorize(palette.get)(neuron_ids)

    plot_mode = "3d"
    navis.plot2d(
        neurons,
        color=colors,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=autoscale,
        soma=False,
        **kwargs,
    )
    # plot_volumes(volumes, ax)
    if plot_mode == "3d":
        ax.azim = azim
        ax.elev = elev
        ax.dist = dist
        if axes_equal:
            set_axes_equal(ax, use_y=use_y, use_x=use_x, use_z=use_z)
    if axis_off:
        ax.axis("off")
    if force_bounds:
        ax.set_xlim3d((-4500, 110000))
        ax.set_ylim3d((-4500, 110000))
    return ax


def plot_volumes(volumes, ax):
    navis.plot2d(volumes, ax=ax, method="3d", autoscale=False)
    for c in ax.collections:
        if isinstance(c, Poly3DCollection):
            c.set_alpha(0.02)


def set_axes_equal(ax, use_x=True, use_y=True, use_z=True):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    # REF: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    dimension_ranges = []
    if use_x:
        dimension_ranges.append(x_range)
    if use_y:
        dimension_ranges.append(y_range)
    if use_z:
        dimension_ranges.append(z_range)

    plot_radius = 1 * max(dimension_ranges)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
