from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
import numpy as np


def fit_bounding_contour(points, s=0, per=1):
    hull = ConvexHull(points)
    boundary_indices = list(hull.vertices)
    boundary_indices.append(boundary_indices[0])
    boundary_points = points[boundary_indices].copy()

    tck, u = splprep(boundary_points.T, u=None, s=s, per=per)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new


def draw_bounding_contour(points, ax=None, color=None, linewidth=2):
    x_new, y_new = fit_bounding_contour(points)
    ax.plot(x_new, y_new, color=color, zorder=-1, linewidth=linewidth, alpha=0.5)
    ax.fill(x_new, y_new, color=color, zorder=-2, alpha=0.1)


def bound_points(points, ax=None, point_data=None, label=None, palette=None):
    if point_data is not None:
        labels = point_data[label]
    else:
        labels = label

    uni_labels = np.unique(labels)
    for group in uni_labels:
        label_points = points[group == labels]
        draw_bounding_contour(label_points, color=palette[group], ax=ax)
