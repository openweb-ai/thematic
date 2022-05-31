"""
Module to do multi-resolution clustering.
"""
import logging
from typing import Iterable, List, Union, cast

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
        member_counts: Union[List[int], np.ndarray] = None,
    ) -> "FastCommunityDetection":
        self.labels_ = np.repeat(-1, len(X))
        cluster_centers = cast(List[int], [])
        clusters = community_detection(X, member_counts=member_counts, **self._kwargs)
        for cluster_id, cluster in enumerate(clusters):
            cluster_centers.append(X[cluster[0]])
            for idx in cluster:
                self.labels_[idx] = cluster_id
        self.cluster_centers_ = np.array(cluster_centers)
        return self

    def fit_predict(
        self, X, y=None, member_counts: Union[List[int], np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(X, y, member_counts=member_counts).labels_


class MultiResCommunityDetection:
    """multi-resolution community detection"""

    methods = {"fast_community_detection": FastCommunityDetection}

    def __init__(
        self,
        params: Iterable[dict],
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self._params = list(params)
        self._factory = FastCommunityDetection
        self.labels_: np.ndarray = np.array([])
        self.cluster_centers_: np.ndarray = np.ndarray([])
        self._logger = logger

    def fit(
        self,
        X,
        y=None,  # pylint: disable=unused-argument
        member_counts: Union[List[int], np.ndarray] = None,
    ) -> "MultiResCommunityDetection":
        self.labels_ = np.repeat(-1, len(X))
        cluster_centers = cast(List[int], [])
        mutable_x = np.array(X)
        indices = np.array(range(len(X)))
        if member_counts is not None:
            mutable_member_counts = np.array(member_counts).flatten()
        else:
            mutable_member_counts = np.ones(len(X))
        cluster_cnt = 0
        for idx, kwargs in enumerate(self._params):
            if len(mutable_x) == 0:
                self._logger.info("all items has been assigned to a cluster")
                break

            model = self._factory(**kwargs)
            self._logger.info("created model #%s: %s", idx, kwargs)

            labels = model.fit_predict(mutable_x, member_counts=mutable_member_counts)
            centers = model.cluster_centers_.tolist()
            self._logger.info("found %s clusters", len(centers))

            cluster_centers += centers
            np.put(
                self.labels_,
                ind=indices[np.argwhere(labels >= 0).reshape(-1)],
                v=labels[labels >= 0] + cluster_cnt,
            )
            cluster_cnt += max(self.labels_) + 1
            mutable_x = mutable_x[labels < 0]
            indices = indices[labels < 0]
            mutable_member_counts = mutable_member_counts[labels < 0]
        self.cluster_center_ = np.array(cluster_centers)
        return self
