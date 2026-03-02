import numpy as np
from scipy.stats import ks_2samp
import pandas as pd

def detect_drift(train_data, test_data, p_threshold=0.05):
    """
    Performs Kolmogorov-Smirnov test to see if our testing data has drifted away 
    from our training data like a teenager ignoring texts :(

    If the p_value is less than our threshold, we scream 'drift_detected = True'.

    Args:
        train_data (pd.DataFrame or dict): Training features where everything was perfect.
        test_data (pd.DataFrame or dict): Testing features that are currently breaking things.
        p_threshold (float): Threshold to decide if drift is real or just our anxiety.

    Returns:
        dict: A report card of KS stats, p-values, and whether drift occurred per feature.
    """
    # If users passed dicts or arrays instead of DataFrames, we force them to be DataFrames
    # because DataFrames are nicely structured and Python gets mad if we don't.
    if not isinstance(train_data, pd.DataFrame):
        train_data = pd.DataFrame(train_data)
    if not isinstance(test_data, pd.DataFrame):
        test_data = pd.DataFrame(test_data)
        
    drift_report = {}
    
    # We iterate over common columns to avoid throwing KeyErrors
    common_cols = [col for col in train_data.columns if col in test_data.columns]
    
    for feature in common_cols:
        # Drop NaNs because KS test hates them
        train_dist = train_data[feature].dropna().values
        test_dist = test_data[feature].dropna().values
        
        # If we have no data left, just skip. We can't evaluate thin air.
        if len(train_dist) == 0 or len(test_dist) == 0:
            continue
            
        # The actual magic. It compares the two statistical distributions.
        ks_stat, p_value = ks_2samp(train_dist, test_dist)
        
        drift_report[feature] = {
            "ks_stat": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < p_threshold)
        }
        
    return drift_report
