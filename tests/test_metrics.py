import numpy as np
from evalforge.metrics import calculate_metrics

def test_calculate_metrics_basic():
    """
    Test basic classification metrics calculation.
    If this fails, math is broken or my code is :(
    """
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]  # One false negative
    
    results = calculate_metrics(y_true, y_pred)
    
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
    
    results = calculate_metrics(y_true, y_pred, y_prob=y_prob)
    
    assert results["roc_auc"] is not None
    assert results["roc_auc"] == 1.0  # Perfect separation

def test_calculate_metrics_edge_cases():
    """
    Test edge cases like zero division where precision/recall might scream at us.
    """
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    
    # Zero division is handled gracefully in the implementation with zero_division=0
    results = calculate_metrics(y_true, y_pred)
    assert results["precision"] == 0.0
    assert results["f1"] == 0.0
