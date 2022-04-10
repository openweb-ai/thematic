"""
Module to do multi-resolution clustering.
"""
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    NamedTuple,
    Sequence,
    Set,
    Tuple,
    Union,
)

from openweb.thematic.third_party import community_detection


class MultiResCluster(NamedTuple):
    threshold: float
    min_community_size: int
    cluster_ids: Tuple[int, ...]


def _normalize_params(
    thresholds: Sequence[float],
    min_community_sizes: Union[Sequence[int], int],
) -> Iterable[Tuple[float, int]]:
    min_community_sizes = list(
        [min_community_sizes] * len(thresholds)
        if isinstance(min_community_sizes, int)
        else min_community_sizes
    )
    len_comm_size = len(min_community_sizes)
    len_thresholds = len(thresholds)
    if len_comm_size > len_thresholds:
        min_community_sizes = min_community_sizes[:len_thresholds]
    elif len_comm_size < len_thresholds:
        min_community_sizes += [min_community_sizes[-1]] * (
            len_thresholds - len_comm_size
        )
    return zip(thresholds, min_community_sizes)


def community_detection_multi_res(
    embeddings: Any,
    thresholds: Sequence[float] = (0.9, 0.8, 0.7, 0.6, 0.5),
    min_community_sizes: Union[Sequence[int], int] = 10,
    init_max_size: int = 1000,
) -> Iterator[MultiResCluster]:
    in_clusters: Set[int] = set()

    for threshold, min_community_size in _normalize_params(
        thresholds, min_community_sizes
    ):
        # get the indexes and embeddings that are not in any clusters yet
        indexes, unclustered_embeddings = zip(
            *[
                (idx, embedding)
                for idx, embedding in enumerate(embeddings)
                if idx not in in_clusters
            ]
        )
        # create a mapping of new indexes to original indexes
        lookup: Dict[int, int] = dict(enumerate(indexes))
        # do community detection
        clusters = community_detection(
            unclustered_embeddings,
            threshold=threshold,
            min_community_size=min_community_size,
            init_max_size=init_max_size,
        )
        for cluster in clusters:
            # get the original indexes
            cluster_ids = tuple((lookup[idx] for idx in cluster))
            # remember which indexes are already clustered
            for idx in cluster_ids:
                in_clusters.add(idx)
            yield MultiResCluster(
                threshold=threshold,
                min_community_size=min_community_size,
                cluster_ids=cluster_ids,
            )
