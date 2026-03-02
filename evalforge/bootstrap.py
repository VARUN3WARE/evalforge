import numpy as np
from evalforge.metrics import calculate_metrics

def compute_bootstrap_ci(y_true, y_pred, y_prob=None, n_iterations=1000, ci_level=95):
    """
    Calculates bootstrap confidence intervals because a single point estimate 
    is just a model's best guess on a good day :)

    We resample with replacement like a chaotic bartender mixing drinks, 
    then we calculate metrics on each sample and take the percentiles based on ci_level.

    Args:
        y_true (list or np.array): The painful truth.
        y_pred (list or np.array): Model's predictions.
        y_prob (list or np.array, optional): Probabilities (we need this for AUC).
        n_iterations (int): How many times we want to abuse our CPU.
        ci_level (float): The confidence level (usually 95, because 100 is arrogant).

    Returns:
        dict: A dictionary of confidence intervals like {"accuracy_ci": (lower, upper)}.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)

    n_samples = len(y_true)
    
    # Store our bootstrapped dreams here
    boot_acc = []
    boot_f1 = []
    boot_auc = []

    for i in range(n_iterations):
        # We sample indices randomly, with replacement. Chaos time.
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        sample_y_true = y_true[indices]
        sample_y_pred = y_pred[indices]
        sample_y_prob = y_prob[indices] if y_prob is not None else None
        
        # Calculate metrics for this specific hallucination of the dataset
        metrics = calculate_metrics(sample_y_true, sample_y_pred, sample_y_prob)
        
        boot_acc.append(metrics["accuracy"])
        boot_f1.append(metrics["f1"])
        if metrics["roc_auc"] is not None:
            boot_auc.append(metrics["roc_auc"])

    # Calculate the quantiles to find our lower and upper bounds
    alpha = (100 - ci_level) / 2.0
    lower_p = alpha
    upper_p = 100 - alpha

    results = {
        "accuracy_ci": (np.percentile(boot_acc, lower_p), np.percentile(boot_acc, upper_p)),
        "f1_ci": (np.percentile(boot_f1, lower_p), np.percentile(boot_f1, upper_p)),
    }
    
    if boot_auc:
        results["auc_ci"] = (np.percentile(boot_auc, lower_p), np.percentile(boot_auc, upper_p))
        
    return results
