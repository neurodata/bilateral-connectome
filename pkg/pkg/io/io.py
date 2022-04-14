import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue as default_glue


def _handle_dirs(pathname, foldername, subfoldername):
    path = Path(pathname)
    if foldername is not None:
        path = path / foldername
        if not os.path.isdir(path):
            os.mkdir(path)
        if subfoldername is not None:
            path = path / subfoldername
            if not os.path.isdir(path):
                os.mkdir(path)
    return path


RESULTS_PATH = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = RESULTS_PATH / "results"

FIG_PATH = RESULTS_PATH / "figs"

OUT_PATH = RESULTS_PATH / "outputs"


def savefig(
    name,
    format="png",
    dpi=300,
    foldername=None,
    subfoldername=None,
    pathname=FIG_PATH,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=False,
    print_out=False,
    formats=["png", "pdf", "svg"],
    **kws,
):
    path = _handle_dirs(pathname, foldername, subfoldername)
    savename = path / str(name)
    for format in formats:
        plt.savefig(
            str(savename) + "." + format,
            format=format,
            facecolor="white",
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            dpi=dpi,
            **kws,
        )
    if print_out:
        print(f"Saved figure to {savename}")


def get_out_dir(foldername=None, subfoldername=None, pathname=OUT_PATH):
    path = _handle_dirs(pathname, foldername, subfoldername)
    return path


def glue(name, var, filename, figure=False, display=False, form=None):
    savename = f"{filename}-{name}"

    if figure:
        savename = "fig:" + savename
    else:
        # JSON
        with open(RESULTS_PATH / "glued_variables.json", "r") as f:
            variables = json.load(f)
        # numpy types are not json serializable
        if isinstance(var, np.generic):
            var = var.item()
        variables[savename] = var
        with open(RESULTS_PATH / "glued_variables.json", "w") as f:
            json.dump(variables, f)

        # TXT
        text = ""
        for key, val in variables.items():
            text += key + " " + str(val) + "\n"
        with open(RESULTS_PATH / "glued_variables.txt", "w") as f:
            f.write(text)

    default_glue(savename, var, display=display)

    if form == "pvalue":
        if var > 0.01:
            var = f"{var:0.2f}"
        else:
            factor = int(np.ceil(np.log10(var)))
            var = r"${<}10^{" + str(factor) + r"}$"
    elif form == "long":
        var = f"{var:,}"
    elif form == "2.0f%":
        var = f"{var*100:2.0f}"
    elif form == ".1f%":
        var = f"{var*100:.1f}"
    elif form == "0.2f":
        var = f"{var:0.2f}"
    elif form == "0.2g":
        var = f"{var:0.2g}"

    if form is not None:
        glue(
            name + "-formatted", var, filename, figure=figure, display=False, form=None
        )
