import numpy as np
import pandas as pd

from scipy.spatial.distance import mahalanobis
from sklearn import preprocessing


def encode_labels(y_train, y_calib, y_test=None):
    """Converts string categorical labels into integers."""

    le = preprocessing.LabelEncoder()
    le.fit(y_train.tolist())

    y_train_l = np.array(le.transform(y_train.tolist()))
    y_calib_l = np.array(le.transform(y_calib.tolist()))
    if y_test is None:
        return le, y_train_l, y_calib_l
    else:
        y_test_l = np.array(le.transform(y_test.tolist()))
        return le, y_train_l, y_calib_l, y_test_l


def get_singlets_alpha(target_data, p, le, classes, add_key="singlets_alpha"):
    predictions = pd.DataFrame(p, columns=classes, index=target_data.obs_names)

    # Single labels under alpha
    labels = predictions[predictions.sum(1) == 1].idxmax(1)
    labels = pd.Series(le.inverse_transform(labels), index=labels.index)

    # Attach single-labels under alpha to target data
    if add_key in target_data.obs.columns:
        raise ValueError(
            f"key {add_key} already exists in .obs. Please provide a key that doesn't already exist."
        )
    target_data.obs.loc[labels.index, add_key] = labels

    return target_data


def calc_p(ncal, ngt, neq, smoothing=False):
    if smoothing:
        return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)


class NoNorm:
    def __init__(self):
        pass

    def fit(self, _):
        pass

    def predict(self, X):
        return np.ones((X.shape[0],), dtype=np.float32)

class CenterDist:
    def __init__(
        self,
        seed: int = 42,
        filter_threshold: float = None,
    ):

        self.filter_threshold = filter_threshold
        self.seed = seed

        np.random.seed(seed)

    def fit(self, X):

        if self.filter_threshold is not None:
            IV_raw = np.linalg.pinv(np.cov(X.T)).T
            center_train_raw = X.mean(axis=0)
            dists_raw = np.array([mahalanobis(x, center_train_raw, VI=IV_raw) for x in X])
            dist_thresh_raw = np.quantile(dists_raw, self.filter_threshold)
            X = X[dists_raw < dist_thresh_raw]

        self.IV = np.linalg.pinv(np.cov(X.T)).T
        self.center = X.mean(axis=0)

    def predict(self, X):
        return np.array([mahalanobis(x, self.center, VI=self.IV) for x in X])
    
class NoErr:
    def __init__(self):
        pass

    def compute(self, y, _):
        return np.ones(y.size, dtype=np.float32)
    
class InvProbSquaredError:
    def __init__(self):
        pass

    def compute(self, y, prediction):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = (1 - prediction[i, int(y_)]) ** 2
        return prob


class MarginErr:
    def __init__(self):
        pass

    def compute(self, y, prediction):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
        return 0.5 - ((prob - prediction.max(axis=1)) / 2)