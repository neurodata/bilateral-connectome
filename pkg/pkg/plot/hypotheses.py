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
        title_text = "Density"
    elif model in ["sbm", "dasbm"]:
        if subscript:
            item1 = r"$B^{(L)}_{ij}$"
            item2 = r"$B^{(R)}_{ij}$"
        else:
            item1 = r"$B^{(L)}$"
            item2 = r"$B^{(R)}$"
        if model == "sbm":
            title_text = "Group connection"
        elif model == "dasbm":
            title_text = "Density-adjusted\ngroup connection"

    text_items = [r"$H_0$:", item1, r"$=$", item2]
    colors = ["black", network_palette["Left"], "black", network_palette["Right"]]
    spaces = [True, True, True]

    if model == "dasbm":
        text_items.insert(3, r"$c$")
        colors.insert(3, "black")
        spaces.insert(3, False)

    texts = multicolor_text(
        x,
        y,
        text_items,
        colors,
        spaces=spaces,
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
        spaces=spaces,
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
        ax.text(
            points[0][0],
            top,
            title_text,
            va="bottom",
            ha="left",
            transform=ax.transData,
            fontsize="large",
        )

    return texts
