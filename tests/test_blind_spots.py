import numpy as np

from evalforge.blind_spots import map_blind_spots


def test_map_blind_spots_identifies_underperforming_cluster():
    """
    Creates one obviously bad region so blind spot detection has something to catch.
    """
    rng = np.random.default_rng(42)

    cluster_good = rng.normal(loc=-2.0, scale=0.3, size=(50, 2))
    cluster_bad = rng.normal(loc=2.0, scale=0.3, size=(50, 2))
    X = np.vstack([cluster_good, cluster_bad])

    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.array([0] * 50 + [0] * 50)

    result = map_blind_spots(X, y_true, y_pred, n_clusters=2, random_state=42)

    assert result["n_clusters"] == 2
    assert "cluster_report" in result
    assert len(result["cluster_report"]) == 2
    assert len(result["blind_spot_clusters"]) >= 1


def test_map_blind_spots_shape_validation():
    """
    Mismatched shapes should fail early instead of producing cryptic misery.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = [0, 1]
    y_pred = [0]

    try:
        map_blind_spots(X, y_true, y_pred, n_clusters=2)
        assert False, "Expected ValueError for mismatched input lengths"
    except ValueError:
        assert True
