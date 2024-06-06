import os
import logging
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import logging

from collections.abc import MutableMapping
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List

import pandas as pd
import torch

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


# transformation of model predictions

def output_collate(
    outputs: List[Any], ignore_keys: List[str] = ["loss", "loss_bin_clas", "loss_adv"]
):
    """Collates list of batched nested dict outputs into single dict of listed samples."""
    out = {}
    for d in outputs:
        for k in ignore_keys:
            d.pop(k, None)
        out = _recursive_dict_collate(d, out)
    out = _recursive_dict_flatten_lists(out)
    return out


def _recursive_dict_collate(d, out):
    for k, v in d.items():
        if isinstance(v, dict):
            if k not in out:
                out[k] = {}
            out[k] = _recursive_dict_collate(v, out[k])
        else:
            if k not in out:
                out[k] = []
            if isinstance(v, torch.Tensor):
                v = [_get_value(v_) for v_ in v.detach().cpu()]
            elif isinstance(v, list):
                pass
            else:
                raise ValueError(f"Unknown type: {type(v)} for {v}")
            out[k].append(v)
    return out


def _get_value(tensor):
    return tensor.item() if torch.numel(tensor) == 1 else tensor


def _recursive_dict_flatten_lists(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _recursive_dict_flatten_lists(v)
        elif isinstance(v, list):
            d[k] = list(chain(*v))
        else:
            raise ValueError(f"{v} should be of type 'dict' or 'list' but is {type(v)}")
    return d

def process_logits(predictions):
    tmp = {}
    for head, v in predictions["x"].items():
        tmp[head] = {}
        logits = torch.stack(v)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        tmp[head]["pred_cls"] = [p.item() for p in preds]
        for i in range(probs.shape[1]):
            tmp[head][f"prob_cls_{i}"] = [p[i].item() for p in probs]
    predictions["x"] = tmp
    return predictions

def transform_output_dict(output):
    output = _flatten_dict_keys(output)
    output = _separate_tensors(output)
    return _to_out_dict(output)


def _flatten_dict_keys(dictionary, parent_key="", separator="|"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten_dict_keys(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _separate_tensors(d):
    nonscalar_tensors = {}
    d_tmp = deepcopy(d)
    for k, v in d_tmp.items():
        if isinstance(v[0], torch.Tensor):
            if v[0].numel() > 1:
                v_tmp = d.pop(k)
                nonscalar_tensors[k] = [v_.numpy() for v_ in v_tmp]
            else:
                print(
                    f"Scalar tensor {v} found for key {k}! All tensors should be nonscalar at this point!"
                )
    return d | {"nonscalar": nonscalar_tensors}


def _to_out_dict(predictions):
    nonscalar = predictions.pop("nonscalar", None)
    return {"scalar": pd.DataFrame(predictions), "nonscalar": nonscalar}