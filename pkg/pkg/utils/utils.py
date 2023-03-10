import warnings

import numpy as np
from beartype.roar import BeartypeDecorHintPepDeprecatedWarning


def set_warnings():
    # warnings.filterwarnings("ignore", category=UserWarning, module="umap")

    # this is currently being thrown on import of graspologic (11/05/2021)
    warnings.filterwarnings(
        "ignore", module="beartype", category=BeartypeDecorHintPepDeprecatedWarning
    )


def get_random_seed(random_state):
    seed = random_state.integers(np.iinfo(np.int32).max)
    return seed
