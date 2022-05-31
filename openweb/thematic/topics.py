"""
Module that implements a multi-level clustering approach to topic modelling.
"""

import numpy as np

from openweb.thematic.clusters import FastCommunityDetection, MultiResCommunityDetection


class SemanticTopicModel:
    """Implement 2-level clustering for topic modelling"""
    def __init__(
        self,
        unit_threshold: float = 0.85,
        unit_minsize: int = 5,
        unit_maxsize: int = 1000,
        cluster_threshold_start: float = 0.65,
        cluster_threshold_end: float = 0.85,
        cluster_threshold_step: float = 0.05,
        cluster_minsize: int = 5,
        cluster_maxsize: int = 1000,
    ):
        self._cluster_params = [
            {
                "threshold": threshold,
                "min_community_size": cluster_minsize,
                "init_max_size": cluster_maxsize,
            }
            for threshold in np.arange(
                cluster_threshold_start,
                cluster_threshold_end + cluster_threshold_step,
                cluster_threshold_step,
            )[::-1]
        ]
        self._unit_kwargs = {
            "threshold": unit_threshold,
            "min_community_size": unit_minsize,
            "init_max_size": unit_maxsize,
        }
        self.unit_cluster_labels: np.ndarray = np.array([])
        self.cluster_labels: np.ndarray = np.array([])
        self.unit_labels: np.ndarray = np.array([])

    def fit(
        self,
        X,
        y=None,  # pylint: disable=unused-argument
    ) -> "SemanticTopicModel":

        cluster_labels = np.repeat(-1, len(X))
        unit_labels = FastCommunityDetection(**self._unit_kwargs).fit_predict(X)
        unit_cluster_labels = np.repeat(-1, len(np.unique(unit_labels)))
        n_units = unit_labels.max() + 1
        if n_units > 0:
            unit_x = np.array(
                [X[unit_labels == i].mean(axis=0) for i in range(n_units)]
            )  # Use unit centroid instead
            unit_member_counts = np.array(
                [len(X[unit_labels == i]) for i in range(n_units)]
            )
            unit_cluster_labels = (
                MultiResCommunityDetection(self._cluster_params)
                .fit(unit_x, member_counts=unit_member_counts)
                .labels_
            )
            np.put(
                cluster_labels,
                ind=np.array(range(len(X)))[unit_labels >= 0],
                v=unit_cluster_labels[unit_labels[unit_labels >= 0]],
            )
        self.unit_labels = unit_labels
        self.unit_cluster_labels = unit_cluster_labels
        self.cluster_labels = cluster_labels
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).cluster_labels
