import numpy as np
from sklearn.linear_model import LogisticRegression

from evalforge.fragility import calculate_adversarial_fragility


def test_calculate_adversarial_fragility_returns_expected_keys():
    """
    Ensures fragility output structure is sane and confidence-inspiring.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)

    result = calculate_adversarial_fragility(
        model=model,
        X=X,
        y_true=y,
        noise_std=0.02,
        mask_fraction=0.05,
        scale_fraction=0.05,
        random_state=42,
    )

    assert "baseline_score" in result
    assert "perturbed_scores" in result
    assert "drops" in result
    assert "average_drop" in result
    assert "fragility_score" in result

    assert 0.0 <= result["fragility_score"] <= 100.0
    assert set(result["perturbed_scores"].keys()) == {
        "gaussian_noise",
        "feature_masking",
        "feature_scaling",
    }
