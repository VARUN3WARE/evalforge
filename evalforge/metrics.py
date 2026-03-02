import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates the foundational classification metrics because accuracy alone 
    is a lie we tell ourselves to sleep at night :)

    We use weighted averages for precision, recall, and f1 to handle class imbalances 
    without throwing a fit or crashing in production.

    Args:
        y_true (list or np.array): The painful truth (ground truth labels).
        y_pred (list or np.array): What the model hallucinated (predicted labels).
        y_prob (list or np.array, optional): Confidence scores. If you don't have them, 
                                             ROC AUC stays None. Sorry :(

    Returns:
        dict: A dictionary of metrics to make you feel either good or terrible about your model.
    """
    # Convert inputs to numpy arrays just in case some rogue developer passes a tuple or a generator
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None
    }
    
    # Only calculate ROC AUC if probabilities were actually provided
    # Don't try to be a hero and infer them from hard labels
    if y_prob is not None:
        try:
            # We assume binary classification for ROC AUC for now to keep things sane
            # Multi-class ROC AUC requires `multi_class` param and one-hot encoding, so we play it safe
            y_prob = np.array(y_prob)
            
            # If y_prob is 2D (like from predict_proba), grab the probability of the positive class
            if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
                
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            # Look, if ROC AUC fails, we just silently swallow it and return None
            # because debugging shapes at 3 AM is the worst kind of pain :(
            # You can print(e) if you're feeling masochistic
            pass
            
    return metrics
