import cv2
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def wasserstein_distance_metric(u, v):
    return wasserstein_distance(
        u_values=np.arange(len(u)), v_values=np.arange(len(v)), u_weights=u, v_weights=v
    )


def change_bag_color_space(patch_bag: np.ndarray, space: int = cv2.COLOR_RGB2HSV) -> np.ndarray:
    if patch_bag.ndim != 4:
        raise ValueError("Patch bag must be of shape (N, C, H, W)")

    if patch_bag.shape[1] != 3:
        raise ValueError("Channel dimension must be 3 (RGB / HSV)")

    return (
        cv2.cvtColor(patch_bag.transpose(0, 2, 3, 1).reshape(-1, 1, 3), space)
        .reshape(patch_bag.shape[0], patch_bag.shape[2], patch_bag.shape[3], 3)
        .transpose(0, 3, 1, 2)
    )


def match_cumulative_cdf(source_img: np.ndarray, template_hist: pd.DataFrame) -> np.ndarray:
    """adapted from skimage.exposure.match_histograms."""
    _, src_unique_indices, src_counts = np.unique(
        source_img.ravel(), return_inverse=True, return_counts=True
    )

    tmpl_values = template_hist.columns.values.astype(np.uint8)
    tmpl_counts = template_hist.values[0].astype(np.uint32)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / sum(src_counts)
    tmpl_quantiles = np.cumsum(tmpl_counts) / sum(tmpl_counts)

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source_img.shape)
