import numpy as np
from sklearn.metrics import accuracy_score


def calculate_adversarial_fragility(
    model,
    X,
    y_true,
    metric_fn=None,
    noise_std=0.01,
    mask_fraction=0.05,
    scale_fraction=0.05,
    random_state=42,
):
    """
    Measures how fragile a model is under tiny feature chaos.

    We poke the input with noise, masking, and scaling, then measure performance drop.
    Less drop means stronger model. More drop means we cry softly into logs :(

    Args:
        model: Trained model with a predict method.
        X (np.array): Feature matrix.
        y_true (list or np.array): Ground truth labels.
        metric_fn (callable, optional): Metric function (y_true, y_pred) -> float.
        noise_std (float): Std for Gaussian noise.
        mask_fraction (float): Fraction of entries to mask to zero.
        scale_fraction (float): Max feature scaling change (+/- fraction).
        random_state (int): Reproducibility seed.

    Returns:
        dict: Baseline score, perturbed scores, drops, average drop, and fragility score (0-100).
    """
    if not hasattr(model, "predict"):
        raise ValueError("model must implement a predict method.")

    X = np.array(X, dtype=float)
    y_true = np.array(y_true)

    if X.shape[0] != len(y_true):
        raise ValueError("X and y_true must have matching number of samples.")

    scorer = metric_fn if metric_fn is not None else accuracy_score
    rng = np.random.default_rng(random_state)

    baseline_pred = model.predict(X)
    baseline_score = float(scorer(y_true, baseline_pred))

    # 1) Gaussian noise attack: tiny static in the signal.
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    X_noise = X + noise

    # 2) Random masking attack: remove a small slice of information.
    X_mask = X.copy()
    n_total = X_mask.size
    n_mask = int(np.ceil(mask_fraction * n_total))
    if n_mask > 0:
        flat_idx = rng.choice(n_total, size=n_mask, replace=False)
        X_mask.reshape(-1)[flat_idx] = 0.0

    # 3) Feature scaling attack: wiggle each feature by +/- scale_fraction.
    X_scale = X.copy()
    scale_vector = rng.uniform(1.0 - scale_fraction, 1.0 + scale_fraction, size=X.shape[1])
    X_scale = X_scale * scale_vector

    perturbed_scores = {
        "gaussian_noise": float(scorer(y_true, model.predict(X_noise))),
        "feature_masking": float(scorer(y_true, model.predict(X_mask))),
        "feature_scaling": float(scorer(y_true, model.predict(X_scale))),
    }

    drops = {name: float(max(0.0, baseline_score - score)) for name, score in perturbed_scores.items()}
    average_drop = float(np.mean(list(drops.values()))) if drops else 0.0

    # 100 means rock-solid, 0 means folds under light breeze.
    fragility_score = float(np.clip(100.0 * (1.0 - average_drop), 0.0, 100.0))

    return {
        "baseline_score": baseline_score,
        "perturbed_scores": perturbed_scores,
        "drops": drops,
        "average_drop": average_drop,
        "fragility_score": fragility_score,
    }
