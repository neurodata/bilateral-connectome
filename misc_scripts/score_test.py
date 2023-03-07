#%% [markdown]
# # Score test
#%%
import datetime
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pkg.data import load_network_palette, load_unmatched
from pkg.io import FIG_PATH, get_environment_variables
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import (
    SmartSVG,
    draw_hypothesis_box,
    networkplot_simple,
    plot_density,
    plot_pvalues,
    set_theme,
    svg_to_pdf,
)
from pkg.stats import erdos_renyi_test, stochastic_block_test
from pkg.utils import get_toy_palette, sample_toy_networks
from svgutils.compose import Figure, Panel, Text

from giskard.plot import merge_axes

_, _, DISPLAY_FIGS = get_environment_variables()


FILENAME = "score_test"

FIG_PATH = FIG_PATH / FILENAME


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)


#%%

network_palette, NETWORK_KEY = load_network_palette()
neutral_color = sns.color_palette("Set2")[2]

GROUP_KEY = "celltype_discrete"

left_adj, left_nodes = load_unmatched(side="left")
right_adj, right_nodes = load_unmatched(side="right")

left_labels = left_nodes[GROUP_KEY]
right_labels = right_nodes[GROUP_KEY]

#%% [markdown]
# ## ER test
#%%

stat, pvalue, misc = erdos_renyi_test(left_adj, right_adj, method="score")
print(pvalue)
glue("er_pvalue", pvalue, form="pvalue")

fig, ax = plot_density(misc, palette=network_palette)
gluefig("densities", fig)


#%% [markdown]
# ## SBM test

#%%

stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="score",
    combine_method="tippett",
)
print(pvalue)
glue("sbm_pvalue", pvalue, form="pvalue")

set_theme(font_scale=1.25)

fig, ax = plot_pvalues(misc)
gluefig("sbm_pvalues", fig)

#%% [markdown]
# ## aSBM test

#%%

stat, pvalue, misc = stochastic_block_test(
    left_adj,
    right_adj,
    labels1=left_labels,
    labels2=right_labels,
    method="score",
    density_adjustment=True,
    combine_method="tippett",
)
print(pvalue)
glue("asbm_pvalue", pvalue, form="pvalue")

fig, ax = plot_pvalues(misc)
gluefig("dasbm_pvalues", fig)

#%%

fontsize = 9

# methods = SmartSVG(FIG_PATH / "kc_minus_methods.svg")
# methods.set_width(200)
# methods.move(10, 15)
# methods_panel = Panel(
#     methods, Text("A) Kenyon cell removal", 0, 10, size=fontsize, weight="bold")
# )

er = SmartSVG(FIG_PATH / "densities.svg")
er.set_width(130)
er.move(30, 20)
er_panel = Panel(er, Text("B) Density test", 0, 10, size=fontsize, weight="bold"))
er_panel.move(methods.width * 0.87, 0)

sbm = SmartSVG(FIG_PATH / "sbm_pvalues.svg")
sbm.set_width(200)
sbm.move(0, 25)
sbm_panel = Panel(
    sbm, Text("C) Group connection test", 0, 10, size=fontsize, weight="bold")
)
sbm_panel.move(0, methods.height * 0.9)

asbm = SmartSVG(FIG_PATH / "dasbm_pvalues.svg")
asbm.set_width(200)
asbm.move(0, 25)
asbm_panel = Panel(
    asbm,
    Text("D) Density-adjusted", 0, 10, size=fontsize, weight="bold"),
    Text("group connection test", 14, 20, size=fontsize, weight="bold"),
)
asbm_panel.move(methods.width * 0.87, methods.height * 0.9)


fig = Figure(
    (methods.width + er.width) * 1.02,
    (methods.height + sbm.height) * 0.95,
    methods_panel,
    er_panel,
    sbm_panel,
    asbm_panel,
)
fig.save(FIG_PATH / "kc_minus_composite.svg")

svg_to_pdf(FIG_PATH / "kc_minus_composite.svg", FIG_PATH / "kc_minus_composite.pdf")

fig

#%% [markdown]
# End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
