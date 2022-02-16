from .manual_colors import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    plot_class_colormap,
    plot_colors,
)
from .theme import set_theme
from .bound import bound_points, fit_bounding_contour, draw_bounding_contour
from .utils import shrink_axis, draw_colors, remove_shared_ax
from .sbm import (
    plot_stochastic_block_probabilities,
    plot_pvalues,
    heatmap_grouped,
    networkplot_grouped,
)
