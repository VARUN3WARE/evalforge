import numpy as np


def detect_confidence_accuracy_mismatch(
    y_true,
    y_pred,
    y_prob,
    confidence_threshold=0.9,
):
    """
    Detects highly confident wrong predictions, aka the model's loud wrong opinions :)

    A mismatch is counted when confidence > threshold AND prediction is incorrect.

    Args:
        y_true (list or np.array): Ground truth labels.
        y_pred (list or np.array): Predicted labels.
        y_prob (list or np.array): Predicted probabilities. Supports:
            - Binary 1D probabilities for positive class.
            - 2D predict_proba-like outputs.
        confidence_threshold (float): Confidence cutoff to flag risky mistakes.

    Returns:
        dict: Summary containing counts, rate, and mismatch indices.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_prob length must match y_true.")

    # Confidence extraction logic with minimal drama:
    # - 1D probabilities are treated as positive-class probabilities (binary case).
    # - 2D probabilities use max probability per sample.
    if y_prob.ndim == 1:
        confidences = np.where(y_pred == 1, y_prob, 1.0 - y_prob)
    elif y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
    else:
        raise ValueError("y_prob must be a 1D or 2D array-like structure.")

    incorrect_mask = y_pred != y_true
    high_conf_mask = confidences > confidence_threshold
    mismatch_mask = incorrect_mask & high_conf_mask
    mismatch_indices = np.where(mismatch_mask)[0]

    mismatch_count = int(np.sum(mismatch_mask))
    total_samples = int(len(y_true))
    mismatch_rate = float(mismatch_count / total_samples) if total_samples else 0.0

    return {
        "total_samples": total_samples,
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_rate,
        "mismatch_indices": mismatch_indices.tolist(),
    }


def detect_regression_mismatch(y_true, y_pred, std_threshold=2.0):
    """
    Detects samples where the model is severely wrong in a regression task.
    Because sometimes a model is not just off, it's living in a different reality :)

    Args:
        y_true (list or np.array): Ground truth continuous values.
        y_pred (list or np.array): Predicted continuous values.
        std_threshold (float): Flag residuals > this many standard deviations of the residuals.

    Returns:
        dict: Summary containing counts, rate, and high-error indices.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    residuals = np.abs(y_true - y_pred)
    std_residuals = np.std(residuals)
    
    # Avoid division by zero if all residuals are the same
    if std_residuals == 0:
        high_error_mask = np.zeros_like(residuals, dtype=bool)
    else:
        high_error_mask = residuals > (std_threshold * std_residuals)
        
    mismatch_indices = np.where(high_error_mask)[0]
    mismatch_count = int(np.sum(high_error_mask))
    total_samples = int(len(y_true))
    mismatch_rate = float(mismatch_count / total_samples) if total_samples else 0.0

    return {
        "total_samples": total_samples,
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_rate,
        "mismatch_indices": mismatch_indices.tolist(),
        "std_residuals": float(std_residuals)
    }
