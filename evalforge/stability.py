import numpy as np


def compute_stability_from_scores(seed_scores, max_std=0.1):
    """
    Converts per-seed performance scores into a stability score (0-100).

    Lower variance across seeds means the model is calm and mature.
    Higher variance means the model has mood swings :)

    Args:
        seed_scores (list or np.array): Performance scores from different seeds.
        max_std (float): Std threshold that maps to a stability floor of 0.

    Returns:
        dict: Mean, variance, std, raw scores, and stability score.
    """
    seed_scores = np.array(seed_scores, dtype=float)
    if seed_scores.size == 0:
        raise ValueError("seed_scores cannot be empty.")
    if max_std <= 0:
        raise ValueError("max_std must be greater than 0.")

    mean_score = float(np.mean(seed_scores))
    variance = float(np.var(seed_scores))
    std = float(np.std(seed_scores))

    normalized_instability = min(std / max_std, 1.0)
    stability_score = float(np.clip(100.0 * (1.0 - normalized_instability), 0.0, 100.0))

    return {
        "seed_scores": seed_scores.tolist(),
        "mean_score": mean_score,
        "variance": variance,
        "std": std,
        "stability_score": stability_score,
    }


def evaluate_seed_stability(train_eval_fn, seeds=None, max_std=0.1):
    """
    Runs a training/evaluation function across multiple seeds and scores stability.

    Args:
        train_eval_fn (callable): Function that takes `seed` and returns a score.
        seeds (list, optional): Seeds to evaluate. Default is [1, 7, 21, 42, 84].
        max_std (float): Std threshold for stability scaling.

    Returns:
        dict: Stability summary including per-seed scores.
    """
    if seeds is None:
        seeds = [1, 7, 21, 42, 84]

    scores = []
    for seed in seeds:
        score = float(train_eval_fn(seed))
        scores.append(score)

    return compute_stability_from_scores(scores, max_std=max_std)
