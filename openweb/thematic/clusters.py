"""
Module to do multi-resolution clustering.
"""
from typing import Iterable, List, cast

import numpy as np

from openweb.thematic.third_party import community_detection


class FastCommunityDetection:
    """sentence_transformers fast community detection wrapped with sklearn interace."""

    def __init__(self, threshold=0.75, min_community_size=10, init_max_size=1000):
        self._kwargs = {
            "threshold": threshold,
            "min_community_size": min_community_size,
            "init_max_size": init_max_size,
        }
        self.labels_: np.ndarray = np.array([])
        self.cluster_centers_: np.ndarray = np.ndarray([])

    def fit(
        self,
        X,
        y=None,  # pylint: disable=unused-argument
    ) -> "FastCommunityDetection":
        self.labels_ = np.repeat(-1, len(X))
        cluster_centers = cast(List[int], [])
        clusters = community_detection(X, **self._kwargs)
        for cluster_id, cluster in enumerate(clusters):
            cluster_centers.append(X[cluster[0]])
            for idx in cluster:
                self.labels_[idx] = cluster_id
        self.cluster_centers_ = np.array(cluster_centers)
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).labels_


class MultiResCommunityDetection:
    """multi-resolution community detection"""

    methods = {"fast_community_detection": FastCommunityDetection}

    def __init__(
        self,
        params: Iterable[dict],
    ):
        self._params = params
        self._factory = FastCommunityDetection
        self.labels_: np.ndarray = np.array([])
        self.cluster_centers_: np.ndarray = np.ndarray([])

    def fit(
        self,
        X,
        y=None,  # pylint: disable=unused-argument
    ) -> "MultiResCommunityDetection":
        self.labels_ = np.repeat(-1, len(X))
        cluster_centers = cast(List[int], [])
        mutable_x = X
        indices = np.array(range(len(X)))
        cluster_cnt = 0
        for kwargs in self._params:
            model = self._factory(**kwargs)
            labels = model.fit_predict(mutable_x)
            cluster_centers += model.cluster_centers_
            cluster_cnt += max(labels) + 1
            mutable_x = mutable_x[labels < 0]
            indices = indices[labels < 0]
            np.put(
                self.labels_,
                ind=indices[np.argwhere(labels > 0).reshape(-1)],
                v=labels[labels > 0] + cluster_cnt,
            )
        return self
