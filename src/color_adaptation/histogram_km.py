from typing import List

import cv2
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric

from .utils import wasserstein_distance_metric, change_bag_color_space, match_cumulative_cdf

class HistogramKM:
    def __init__(
        self,
        hist_data,
        n_clusters: int = 8,
        tolerance: float = 0.1,
    ) -> None:

        self.n_clusters = n_clusters
        self.tolerance = tolerance

        self.metric = distance_metric(
            type_metric.USER_DEFINED, func=wasserstein_distance_metric
        )

        self.all_histos = hist_data

    def fit(self, ids: List[str]):
        fit_histos = self.all_histos.loc[ids, :]

        print(f"Computing {self.n_clusters} kmeans clusters...")

        self.py_km = kmeans(
            fit_histos,
            fit_histos[: self.n_clusters],
            metric=self.metric,
            tolerance=self.tolerance,
            ccore=True,
        )
        self.py_km.process()
        self.pyCenters = self.py_km.get_centers()

        print(f"Computing nearest cluster for all {len(self.all_histos)} samples...")
        self.all_clusters = pd.DataFrame(
            {"cluster": self.py_km.predict(self.all_histos)}, index=self.all_histos.index
        ).assign(is_fit=lambda df: df.index.isin(ids))

    def match_histograms(self, patch_bag: np.ndarray, sample_id: str) -> np.ndarray:
        cluster_target = self.all_clusters["cluster"].loc[sample_id]
        hist_target = pd.DataFrame(
            [self.pyCenters[cluster_target]], columns=self.all_histos.columns
        )

        print(f"Matching histogram of sample {sample_id} with cluster {cluster_target}")

        patch_bag = change_bag_color_space(patch_bag, space=cv2.COLOR_RGB2HSV)

        for i, c in enumerate(self.all_histos.columns.get_level_values(0).unique()):
            img_source = patch_bag[:, i, :, :]
            patch_bag[:, i, :, :] = match_cumulative_cdf(img_source, hist_target[c])

        patch_bag = change_bag_color_space(patch_bag, space=cv2.COLOR_HSV2RGB)

        # get id of a sample in the fit data belonging to the same cluster (for plotting)
        fit_cluster_id = (
            self.all_clusters[
                (self.all_clusters["cluster"] == cluster_target) & (self.all_clusters["is_fit"])
            ]
            .sample(1)
            .index[0]
        )

        return patch_bag, fit_cluster_id
