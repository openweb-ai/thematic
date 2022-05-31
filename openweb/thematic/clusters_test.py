import collections
from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from openweb.thematic.clusters import MultiResCommunityDetection


@pytest.fixture
def mocked_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_blobs(  # type: ignore
        n_samples=1000,
        centers=25,
        cluster_std=1,
        n_features=100,
        random_state=42,
        return_centers=True,
    )


def test_MultiResCommunityDetection(
    mocked_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """
    Trivial test on the clustering on a symmetrical dataset.
    """
    params = [
        {
            "threshold": 0.95,
            "min_community_size": 15,
            "init_max_size": 1000,
        }
    ]
    X, _y, centers = mocked_data
    model = MultiResCommunityDetection(params).fit(X)

    assert len(model.cluster_centers_) == len(centers), "same number of clusters found"
    assert (
        max(model.labels_) == len(model.cluster_centers_) - 1
    ), "id of clusters should not be more than number of clusters"

    assert collections.Counter(model.labels_) == collections.Counter(
        _y
    ), "each cluster should have the same number of items"


def test_normalize_params():

    ...
