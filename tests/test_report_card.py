from evalforge.report_card import generate_report_card

def test_generate_report_card_excellent():
    """
    Test generating a glowing report card for a perfect model.
    A rare breed :)
    """
    health_score_data = {
        "health_score": 95.5,
        "component_scores": {"accuracy": 96.0, "fragility": 100.0}
    }
    
    report = generate_report_card(health_score_data)
    
    assert "95.5 (Excellent)" in report
    assert "Ship it. It's beautiful." in report
    assert "Base Accuracy: 0.96" in report

def test_generate_report_card_high_risk():
    """
    Test generating a terrible report card for a terrible model.
    The kind of model that keeps you up at night :(
    """
    health_score_data = {
        "health_score": 45.0,
        "component_scores": {"accuracy": 50.0, "fragility": 10.0}
    }
    mismatch_data = {
        "mismatch_count": 50,
        "mismatch_rate": 0.20
    }
    drift_data = {
        "feature_1": {"drift_detected": True},
        "feature_2": {"drift_detected": True}
    }
    fragility_data = {
        "average_drop": 0.45
    }
    
    report = generate_report_card(health_score_data, mismatch_data, drift_data, fragility_data)
    
    assert "45.0 (High Risk)" in report
    assert "Base accuracy is critically low (50.0%)" in report
    assert "50 high-confidence errors" in report
    assert "Drift Detected: 2" in report
    assert "Fragility Drop: 45.0%" in report
    
    assert "- Enhance feature engineering" in report
    assert "- Apply temperature scaling" in report
    
def test_generate_report_card_missing_data():
    """
    Test that the report card doesn't crash when optional data is missing.
    """
    health_score_data = {
        "health_score": 85.0,
        "component_scores": {"accuracy": 85.0}
    }
    
    report = generate_report_card(health_score_data)
    
    assert "85.0 (Good)" in report
    assert "Base Accuracy: 0.85" in report
    assert "Drift Detected" not in report
    assert "Fragility Drop" not in report
