from .bound import bound_points, draw_bounding_contour, fit_bounding_contour
from .er import plot_density
from .hypotheses import draw_hypothesis_box
from .layout import networkplot_simple
from .manual_colors import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    plot_class_colormap,
    plot_colors,
)
from .neuron import simple_plot_neurons
from .sbm import (
    compare_probability_row,
    heatmap_grouped,
    plot_pvalues,
    plot_stochastic_block_probabilities,
)
from .svg import SmartSVG
from .theme import set_theme
from .utils import (
    bound_texts,
    draw_colors,
    get_text_points,
    get_text_width,
    get_texts_points,
    make_sequential_colormap,
    multicolor_text,
    nice_text,
    rainbowarrow,
    remove_shared_ax,
    shrink_axis,
)
