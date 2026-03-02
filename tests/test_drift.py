import pandas as pd
import numpy as np
from evalforge.drift import detect_drift

def test_detect_drift_no_drift():
    """
    Test KS Drift test when there is no drift.
    A quiet day in production :)
    """
    np.random.seed(42)
    
    train_data = pd.DataFrame({"feature_A": np.random.normal(0, 1, 1000)})
    test_data = pd.DataFrame({"feature_A": np.random.normal(0, 1, 100)})
    
    report = detect_drift(train_data, test_data)
    
    assert "feature_A" in report
    assert not report["feature_A"]["drift_detected"]
    
def test_detect_drift_with_drift():
    """
    Test KS Drift test when there IS drift.
    Something went terribly wrong in the real world :(
    """
    np.random.seed(42)
    
    train_data = pd.DataFrame({"feature_A": np.random.normal(0, 1, 1000)})
    # Shift the mean to 2
    test_data = pd.DataFrame({"feature_A": np.random.normal(2, 1, 100)})
    
    report = detect_drift(train_data, test_data)
    
    assert "feature_A" in report
    assert report["feature_A"]["drift_detected"]

def test_detect_drift_edge_cases():
    """
    Handle missing columns without crying.
    """
    train_data = pd.DataFrame({"feat_1": [1, 2], "feat_2": [3, 4]})
    test_data = pd.DataFrame({"feat_1": [1, 2], "feat_3": [5, 6]}) # Missing feat_2
    
    report = detect_drift(train_data, test_data)
    
    assert "feat_1" in report
    assert "feat_2" not in report
    assert "feat_3" not in report
