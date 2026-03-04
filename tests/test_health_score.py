from evalforge.health_score import compute_health_score

def test_compute_health_score_all_components():
    """
    Test calculating the health score when the model provides everything.
    A rare scenario where everything works first try :)
    """
    # Simulate a decent model
    accuracy = 0.90 # 90%
    mismatch_rate = 0.10 # 10% mismatch -> 90% calib
    fragility = 80.0 # 80/100
    drift_report = {
        "f1": {"drift_detected": False},
        "f2": {"drift_detected": True}, # 1/2 drifted -> 50% drift score
    }
    stability = 85.0 # 85/100
    
    # Calculation:
    # Acc: 90 * 0.40 = 36
    # Calib: 90 * 0.15 = 13.5
    # Frag: 80 * 0.20 = 16
    # Drift: 50 * 0.15 = 7.5
    # Stab: 85 * 0.10 = 8.5
    # Total: 36 + 13.5 + 16 + 7.5 + 8.5 = 81.5
    
    result = compute_health_score(accuracy, mismatch_rate, fragility, drift_report, stability)
    assert "health_score" in result
    assert result["health_score"] == 81.5
    
    assert "accuracy" in result["component_scores"]
    assert result["component_scores"]["accuracy"] == 90.0
    assert result["component_scores"]["drift"] == 50.0

def test_compute_health_score_missing_components():
    """
    Test calculating the health score when some metrics are too lazy to show up.
    The weights should dynamically re-distribute.
    """
    # Only accuracy and stability (0.4 + 0.1 = 0.5 total weight)
    # They should each be doubled in importance.
    accuracy = 0.80 # 80 score -> weight 0.8
    stability = 60.0 # 60 score -> weight 0.2
    
    result = compute_health_score(accuracy_metric=accuracy, stability_score=stability)
    
    # Expected: (80 * 0.8) + (60 * 0.2) = 64 + 12 = 76.0
    assert result["health_score"] == 76.0
    assert "drift" not in result["component_scores"]
