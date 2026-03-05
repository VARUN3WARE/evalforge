def compute_health_score(
    accuracy_metric,
    mismatch_rate=None,
    fragility_score=None,
    drift_report=None,
    stability_score=None,
    bias_penalty=0.0
):
    """
    Computes the final Model Health Score (0-100).
    Because accuracy alone is a metric that hides its flaws under a trench coat :)
    
    Weights:
    - Base accuracy: 40%
    - Calibration / mismatch: 15%
    - Fragility score: 20%
    - Drift score: 15%
    - Stability score: 10%
    
    If an optional component is missing, its weight is distributed proportionally 
    to the available components so we don't unfairly penalize the model for our laziness.

    Args:
        accuracy_metric (float): Base accuracy (0.0 to 1.0).
        mismatch_rate (float, optional): Rate of highly confident errors (0.0 to 1.0).
        fragility_score (float, optional): Fragility score (0 to 100).
        drift_report (dict, optional): Drift report dictionary from detect_drift.
        stability_score (float, optional): Stability score (0 to 100).

    Returns:
        dict: Final health score and individual component scores (all 0-100).
    """
    # 1. Normalize everything to 0-100
    acc_score = accuracy_metric * 100.0 if accuracy_metric is not None else 0.0
    
    calib_score = None
    if mismatch_rate is not None:
        calib_score = (1.0 - mismatch_rate) * 100.0
        
    drift_score = None
    if drift_report is not None:
        total_features = len(drift_report)
        if total_features > 0:
            drifted_features = sum(1 for feat in drift_report.values() if feat.get("drift_detected", False))
            drift_score = (1.0 - (drifted_features / total_features)) * 100.0
        else:
            drift_score = 100.0
            
    # Base Weights defined in plan
    weights = {
        "accuracy": 0.40,
        "calibration": 0.15,
        "fragility": 0.20,
        "drift": 0.15,
        "stability": 0.10
    }
    
    components = {
        "accuracy": acc_score,
        "calibration": calib_score,
        "fragility": fragility_score,
        "drift": drift_score,
        "stability": stability_score
    }
    
    active_weights = 0.0
    final_score = 0.0
    component_scores = {}
    
    # Calculate active weights and individual component values
    for name, value in components.items():
        if value is not None:
            active_weights += weights[name]
            component_scores[name] = max(0.0, min(100.0, value))
            
    if active_weights == 0.0:
        return {"health_score": 0.0, "component_scores": {}}

    # Re-distribute weights and calculate final score
    for name in component_scores:
        proportional_weight = weights[name] / active_weights
        final_score += component_scores[name] * proportional_weight
        
    final_score = max(0.0, final_score - bias_penalty)
        
    return {
        "health_score": round(final_score, 2),
        "component_scores": component_scores
    }
