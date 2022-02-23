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
):

    network_palette, _ = load_network_palette()

    if model == "er":
        item1 = r"$p^{(L)}$"
        item2 = r"$p^{(R)}$"
    elif model in ["sbm", "asbm"]:
        if subscript:
            item1 = r"$B^{(L)}_{ij}$"
            item2 = r"$B^{(R)}_{ij}$"
        else:
            item1 = r"$B^{(L)}$"
            item2 = r"$B^{(R)}$"

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
    bound_texts(
        texts,
        ax=ax,
        xpad=xpad,
        ypad=ypad,
        facecolor="white",
        edgecolor="lightgrey",
    )
    return texts
