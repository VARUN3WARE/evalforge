import pandas as pd
import numpy as np
from evalforge.fairness import evaluate_fairness

def test_evaluate_fairness_unbiased():
    df = pd.DataFrame({"gender": ["M", "M", "F", "F"]})
    y_pred = np.array([1, 0, 1, 0])
    
    results = evaluate_fairness(df, y_pred, "gender")
    assert results["bias_detected"] is False
    assert results["bias_penalty"] == 0.0

def test_evaluate_fairness_biased():
    df = pd.DataFrame({"gender": ["M", "M", "F", "F"]})
    y_pred = np.array([1, 1, 0, 0]) # 100% for M, 0% for F
    
    results = evaluate_fairness(df, y_pred, "gender")
    assert results["bias_detected"] is True
    assert results["bias_penalty"] > 20.0
    assert "group_rates" in results
