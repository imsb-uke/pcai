import os
import logging
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import logging

def plot_mask_overlay(mask, cmap=None, ax=None, plot_colorbar=False, **imshow_kwargs):

    ax = ax or plt.gca()

    # create cmap
    if cmap is None:
        color_dict = {
            -1: (0, 0, 0, 0),  # -1 = transparent
            np.nan: (0, 0, 0, 0),  # nan = transparent
            0: (0, 1, 0, 1),  # 0 = green
            1: (1, 0, 0, 1),  # 1 = red
        }

        cmap = matplotlib.colors.ListedColormap(
            [v for k, v in color_dict.items() if k in np.unique(mask)]
        )

    i = ax.imshow(mask, cmap=cmap, **imshow_kwargs)

    if plot_colorbar:
        plt.colorbar(i, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    return ax


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def append_suffix_before_ext(filename, suffix):
    return os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1]


def get_printlogger(name=__name__) -> logging.Logger:
    import logging
    from importlib import reload

    reload(logging)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG, datefmt="%I:%M:%S")
    return get_pylogger(name)
