from pathlib import Path
from typing import List

import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import (
    calc_p,
    encode_labels,
    get_singlets_alpha,
    NoErr,
    CenterDist,
)


class ConformalPredictor:
    def __init__(
        self,
        alpha: float = 0.2,
        base_error = None,
        normalizer = None,
    ):
        
        self.alpha = alpha
        self.base_error = base_error or NoErr()
        self.normalizer = normalizer or CenterDist(filter_threshold=0.95)

    def fit(self, data: ad.AnnData):
        self.train_data, self.calib_data, self.target_data = (
            data[data.obs["cp_split"] == "train"],
            data[data.obs["cp_split"] == "calib"],
            data[data.obs["cp_split"] == "target"],
        )

        self.X_train, self.X_calib, self.X_test, y_train, y_calib, y_test = (
            self.train_data.X,
            self.calib_data.X,
            self.target_data.X,
            self.train_data.obs["target"],
            self.calib_data.obs["target"],
            self.target_data.obs["target"],
        )

        self.le, self.y_train_l, self.y_calib_l, self.y_test_l = encode_labels(
            y_train, y_calib, y_test
        )

        self.err_scores = {}

        self.probs_calib = self.calib_data.obs[["prob_cls_0", "prob_cls_1"]].to_numpy()
        self.probs_test = self.target_data.obs[["prob_cls_0", "prob_cls_1"]].to_numpy()

        self.normalizer.fit(self.X_train)

        self.normalizer_errors_calib = self.normalizer.predict(self.X_calib)

        self.base_errors_calib = self.base_error.compute(self.y_calib_l, self.probs_calib)
        for label in np.unique(self.y_calib_l):
            label_idxs = self.y_calib_l == label
            base_errors = self.base_errors_calib[label_idxs]
            normalizer_errors = self.normalizer_errors_calib[label_idxs]
            self.err_scores[label] = np.sort(base_errors * normalizer_errors)[::-1]

        self.normalizer_errors_test = self.normalizer.predict(self.X_test)

        ncal_ngt_neq = self._get_ncal_ngt_neq(
            self.X_test, self.probs_test, self.normalizer_errors_test
        )

        predictions = np.zeros((self.X_test.shape[0], np.unique(self.y_calib_l).shape[0]))
        predictions_cc = predictions.copy()

        for i in range(np.unique(self.y_calib_l).shape[0]):
            for j in range(self.X_test.shape[0]):
                predictions[j, i] = predictions_cc[j, i] = calc_p(
                    ncal_ngt_neq[j, i, 0], ncal_ngt_neq[j, i, 1], ncal_ngt_neq[j, i, 2]
                )
                predictions[j, i] = predictions[j, i] > self.alpha

        self.target_data = get_singlets_alpha(
            self.target_data.copy(),
            predictions,
            self.le,
            [0, 1],
            add_key="singlets_alpha",
        )

        confidence, credibility, likely_label = self._get_cred_conf_ll(predictions_cc)

        self.target_data.obs["singlets_base"] = likely_label
        self.target_data.obs["confidence"] = confidence
        self.target_data.obs["credibility"] = credibility

    def force_predict_class_rate(self, threshold=0.75, on="credibility"):

        ncal_ngt_neq = self._get_ncal_ngt_neq(
            self.X_calib, self.probs_calib, self.normalizer_errors_calib
        )

        predictions_cc_calib = np.zeros(
            (self.X_calib.shape[0], np.unique(self.y_calib_l).shape[0])
        )

        for i in range(np.unique(self.y_calib_l).shape[0]):
            for j in range(self.X_calib.shape[0]):
                predictions_cc_calib[j, i] = calc_p(
                    ncal_ngt_neq[j, i, 0], ncal_ngt_neq[j, i, 1], ncal_ngt_neq[j, i, 2]
                )

        confidence, credibility, _ = self._get_cred_conf_ll(predictions_cc_calib)

        if on == "confidence":
            c_measure = confidence
        elif on == "credibility":
            c_measure = credibility
        else:
            raise ValueError(f"Unknown thresholding method {on}")

        c_thr = np.quantile(c_measure, q=1 - threshold)
        clf_rate_calib = (c_measure > c_thr).sum() / c_measure.shape[0]

        print(
            f"{on} threshold {c_thr}. Classification rate of calibration set: {clf_rate_calib: .2f}"
        )

        idxs_assigned = np.where(self.target_data.obs[on] > c_thr)[0]

        return self._add_assignment(self.target_data.copy(), idxs_assigned).obs

    def force_predict_error_rate(self, threshold=0.2, on="confidence"):

        if on not in ("confidence", "credibility"):
            raise ValueError(f"Unknown thresholding method {on}")

        n_examples = self.target_data.shape[0]

        kj = n_examples * (1 - self.target_data.obs[on])
        idxs_assigned = np.where(kj < int(n_examples * threshold))[0]

        return self._add_assignment(self.target_data.copy(), idxs_assigned).obs

    def _get_ncal_ngt_neq(self, X, probs, norm_errors):
        ncal_ngt_neq = np.zeros((X.shape[0], np.unique(self.y_calib_l).shape[0], 3))

        for label in np.unique(self.y_calib_l):
            base_errors = self.base_error.compute(np.array([label] * X.shape[0]), probs)
            err_scores_tmp = base_errors * norm_errors

            for i in range(err_scores_tmp.shape[0]):
                left_count = np.searchsorted(
                    self.err_scores[label][::-1], err_scores_tmp[i], "left"
                )
                right_count = np.searchsorted(
                    self.err_scores[label][::-1], err_scores_tmp[i], "right"
                )

                ncal_ngt_neq[i, label, 0] = self.err_scores[label][::-1].size
                ncal_ngt_neq[i, label, 1] = self.err_scores[label][::-1].size - right_count
                ncal_ngt_neq[i, label, 2] = right_count - left_count

        return ncal_ngt_neq

    @staticmethod
    def _get_cred_conf_ll(predictions_cc):
        likely_label = predictions_cc.argmax(axis=1)
        credibility = predictions_cc.max(axis=1)
        for i, idx in enumerate(likely_label):
            predictions_cc[i, idx] = -np.inf
        confidence = 1 - predictions_cc.max(axis=1)
        return confidence, credibility, likely_label

    @staticmethod
    def _add_assignment(target_data, idxs_assigned):
        target_data.obs["assignment"] = 0
        target_data.obs.loc[target_data.obs.index[idxs_assigned], "assignment"] = 1
        target_data.obs["assignment"] = target_data.obs["assignment"].astype("category")
        return target_data

    @staticmethod
    def input_transform(
        preds_base,
        preds_targ,
        target_key = "y|clas",
        features_key = "model_meta|features",
        split_key = "data_meta|split",
        domain_key_base = "data_meta|domain",
        domain_key_target = "data_meta|domain",
        prob_clas_0_key = "x|clas|prob_cls_0",
        prob_clas_1_key = "x|clas|prob_cls_1",
        train_domains: List[str] = None,
        calib_domains: List[str] = None,
        target_domains: List[str] = None,
    ):
        """Transform PCAI model predictions into adata object for conformal prediction."""

        data_base = ConformalPredictor._to_adata(
            preds_base,
            target_key,
            features_key,
            split_key,
            prob_clas_0_key,
            prob_clas_1_key,
            domain_key_base,
        )
        data_targ = ConformalPredictor._to_adata(
            preds_targ,
            target_key,
            features_key,
            split_key,
            prob_clas_0_key,
            prob_clas_1_key,
            domain_key_target,
        )

        return ConformalPredictor._split_data(
            data_base,
            data_targ,
            train_domains = train_domains,
            calib_domains = calib_domains,
            target_domains = target_domains,
        )

    @staticmethod
    def _to_adata(
        preds,
        target_key,
        features_key,
        split_key,
        prob_clas_0_key,
        prob_clas_1_key,
        domain_key=None,
    ):

        cols = {
            target_key: "target",
            split_key: "split",
            prob_clas_0_key: "prob_cls_0",
            prob_clas_1_key: "prob_cls_1",
        }

        if domain_key is not None:
            cols[domain_key] = "domain"

        features = np.stack(preds["nonscalar"][features_key])
        obs = (
            preds["scalar"]
            .loc[
                :,
                cols.keys(),
            ]
            .rename(columns=cols)
            .assign(index=lambda df: df.index.astype(str))
            .set_index("index", drop=True)
            .rename_axis(None, axis=0)
        )
        return ad.AnnData(features, obs=obs)

    @staticmethod
    def _split_data(
        data_base,
        data_targ,
        train_domains: List[str] = None,
        calib_domains: List[str] = None,
        target_domains: List[str] = None,
    ):
        
        train_data = ConformalPredictor._get_split_data(data_base, "train", train_domains)
        calib_data = ConformalPredictor._get_split_data(data_base, "val", calib_domains)
        target_data = ConformalPredictor._get_split_data(data_targ, "test", target_domains)

        print(
            f"Created CP data with {train_data.shape[0]} train, {calib_data.shape[0]} calib and {target_data.shape[0]} target samples."
        )

        train_data.obs["cp_split"] = "train"
        calib_data.obs["cp_split"] = "calib"
        target_data.obs["cp_split"] = "target"

        return ad.concat([train_data, calib_data, target_data], join="inner", axis=0)

    @staticmethod
    def _get_split_data(data, split, domains):
        data_out = data[data.obs["split"] == split]
        if domains is not None:
            data_out = data_out[data_out.obs["domain"].isin(domains)]
        return data_out

    @staticmethod
    def output_transform(cp_preds):
        """Transform conformal prediction results into dataframe for further analysis."""

        return (
            cp_preds.loc[
                :,
                [
                    "cp_split",
                    "singlets_alpha",
                    "singlets_base",
                    "confidence",
                    "credibility",
                    "assignment",
                ],
            ]
            .assign(index=lambda df: df.index.astype(int))
            .set_index("index", drop=True)
            .rename_axis(None, axis=0)
        )
