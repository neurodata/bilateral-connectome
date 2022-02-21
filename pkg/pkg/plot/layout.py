import networkx as nx
from graspologic.plot import networkplot
import seaborn as sns
from .bound import bound_points


def networkplot_simple(A, node_data, palette=None, ax=None, group=False):
    g = nx.from_numpy_array(A)
    pos = nx.kamada_kawai_layout(g)
    node_data["x"] = [pos[node][0] for node in node_data.index]
    node_data["y"] = [pos[node][1] for node in node_data.index]
    if group:
        node_hue = "labels"
    else:
        node_hue = None
    networkplot(
        A,
        node_data=node_data,
        node_hue=node_hue,
        x="x",
        y="y",
        edge_linewidth=1.0,
        palette=palette,
        node_sizes=(20, 200),
        node_kws=dict(
            linewidth=1, edgecolor="black", color=sns.color_palette("Set2")[2]
        ),
        node_alpha=1.0,
        edge_kws=dict(color="black"),
        ax=ax,
    )
    if group:
        bound_points(
            node_data[["x", "y"]].values,
            point_data=node_data,
            ax=ax,
            label="labels",
            palette=palette,
        )
    ax.set(xlabel="", ylabel="")
    ax.spines[["left", "bottom"]].set_visible(False)
    return node_data
