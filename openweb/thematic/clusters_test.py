from openweb.thematic.clusters import _normalize_params


def test_normalize_params():

    results = tuple(
        _normalize_params(thresholds=[0.9, 0.5, 0.3], min_community_sizes=10)
    )
    assert results == ((0.9, 10), (0.5, 10), (0.3, 10))

    results = tuple(
        _normalize_params(thresholds=[0.9, 0.5, 0.3], min_community_sizes=[10, 15])
    )
    assert results == ((0.9, 10), (0.5, 15), (0.3, 15))
