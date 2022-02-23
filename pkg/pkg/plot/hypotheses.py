from ..data import load_network_palette
from .utils import bound_texts, multicolor_text


def draw_hypothesis_box(
    model,
    x,
    y,
    yskip=0.15,
    ax=None,
    xpad=0.03,
    ypad=0.01,
    fontsize="large",
    subscript=False,
    title=False,
):

    network_palette, _ = load_network_palette()

    if model == "er":
        item1 = r"$p^{(L)}$"
        item2 = r"$p^{(R)}$"
        title_text = "ER"
    elif model in ["sbm", "asbm"]:
        if subscript:
            item1 = r"$B^{(L)}_{ij}$"
            item2 = r"$B^{(R)}_{ij}$"
        else:
            item1 = r"$B^{(L)}$"
            item2 = r"$B^{(R)}$"
        if model == "sbm":
            title_text = "SBM"
        elif model == "asbm":
            title_text = "aSBM"

    text_items = [r"$H_0$:", item1, r"$=$", item2]
    colors = ["black", network_palette["Left"], "black", network_palette["Right"]]

    if model == "asbm":
        text_items.insert(3, r"$c$")
        colors.insert(3, "black")

    texts = multicolor_text(
        x,
        y,
        text_items,
        colors,
        fontsize=fontsize,
        ax=ax,
    )

    text_items[0] = r"$H_A$:"
    text_items[2] = r"$\neq$"

    texts += multicolor_text(
        x,
        y - yskip,
        text_items,
        colors,
        fontsize=fontsize,
        ax=ax,
    )
    rect = bound_texts(
        texts,
        ax=ax,
        xpad=xpad,
        ypad=ypad,
        facecolor="white",
        edgecolor="lightgrey",
    )
    if title:
        points = rect.get_bbox().get_points()
        top = points[1][1]
        mid = (points[1][0] - points[0][0]) / 2
        print(points)
        ax.text(
            points[0][0],
            top,
            title_text,
            va="bottom",
            ha="left",
            transform=ax.transData,
            fontsize='large'
        )

    return texts
