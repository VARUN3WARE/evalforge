import numpy as np
from evalforge.metrics import calculate_metrics

def test_calculate_metrics_basic():
    """
    Test basic classification metrics calculation.
    If this fails, math is broken or my code is :(
    """
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]  # One false negative
    
    results = calculate_metrics(y_true, y_pred, task_type="classification")
    
    assert "accuracy" in results
    assert "f1" in results
    assert results["accuracy"] > 0.8
    assert results["roc_auc"] is None  # We didn't pass probabilities

def test_calculate_metrics_with_prob():
    """
    Test classification metrics with probability scores for ROC AUC.
    Because sometimes models are confidently wrong, and we need to know :)
    """
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.1, 0.9, 0.2, 0.8]
    
    results = calculate_metrics(y_true, y_pred, y_prob=y_prob, task_type="classification")
    
    assert results["roc_auc"] is not None
    assert results["roc_auc"] == 1.0  # Perfect separation

def test_calculate_metrics_edge_cases():
    """
    Test edge cases like zero division where precision/recall might scream at us.
    """
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    
    # Zero division is handled gracefully in the implementation with zero_division=0
    results = calculate_metrics(y_true, y_pred, task_type="classification")
    assert results["precision"] == 0.0
    assert results["f1"] == 0.0

def test_calculate_metrics_regression():
    """
    Test regression metrics (R2, MAE, RMSE) for continuous predictions.
    """
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    
    results = calculate_metrics(y_true, y_pred, task_type="regression")
    
    assert "r2_score" in results
    assert "mae" in results
    assert "rmse" in results
    
    assert results["mae"] == 0.5  # (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4 = 2.0 / 4 = 0.5
    assert results["r2_score"] > 0.9 # Should be highly accurate
