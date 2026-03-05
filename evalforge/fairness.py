import pandas as pd
import numpy as np

def evaluate_fairness(df, y_pred, sensitive_col):
    """
    Evaluates algorithmic bias by checking Demographic Parity.
    Because deploying a discriminatory model is a great way to meet lawyers :)
    
    Args:
        df (pd.DataFrame): The test dataset containing the sensitive attribute.
        y_pred (np.array): The model's predictions.
        sensitive_col (str): The column name to check for bias (e.g., 'gender', 'age').
        
    Returns:
        dict: Fairness metrics including the calculated 'bias_penalty'.
    """
    if sensitive_col not in df.columns:
        # If the column isn't there, we can't test it.
        return {"bias_detected": False, "bias_penalty": 0.0, "message": f"{sensitive_col} not found"}
        
    df_eval = df.copy()
    df_eval['prediction'] = y_pred
    
    # Calculate positive prediction rate per group
    # We assume '1' is the advantageous outcome for now.
    group_rates = df_eval.groupby(sensitive_col)['prediction'].mean().to_dict()
    
    if len(group_rates) < 2:
        return {"bias_detected": False, "bias_penalty": 0.0, "message": "Only one group present in sensitive col."}
        
    max_rate = max(group_rates.values())
    min_rate = min(group_rates.values())
    
    # Disparate Impact (Ratio)
    # If the min_rate / max_rate is less than the 80% rule (0.8), it's considered biased in many legal frameworks.
    disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
    
    # Difference
    max_diff = max_rate - min_rate
    
    # Compute penalty: if max_diff > 0.1, we start penalizing.
    # Max penalty is 30 points if the difference is 1.0.
    bias_penalty = 0.0
    if max_diff > 0.1:
        bias_penalty = min(30.0, (max_diff - 0.1) * 30.0)
        
    bias_detected = bias_penalty > 0
    
    return {
        "bias_detected": bias_detected,
        "disparate_impact": disparate_impact,
        "max_difference": max_diff,
        "bias_penalty": bias_penalty,
        "group_rates": group_rates
    }
