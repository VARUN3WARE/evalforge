from evalforge.stability import compute_stability_from_scores, evaluate_seed_stability


def test_compute_stability_from_scores_high_stability():
    """
    Tiny variance should produce high stability because life is finally peaceful.
    """
    scores = [0.90, 0.91, 0.89, 0.90, 0.905]
    result = compute_stability_from_scores(scores, max_std=0.1)

    assert "stability_score" in result
    assert result["std"] < 0.02
    assert result["stability_score"] > 80


def test_evaluate_seed_stability_runs_callback_for_each_seed():
    """
    Checks the multi-seed runner without training a giant model and melting laptops.
    """

    def fake_train_eval(seed):
        return 0.8 + (seed % 3) * 0.01

    result = evaluate_seed_stability(fake_train_eval, seeds=[1, 2, 3, 4, 5], max_std=0.1)

    assert len(result["seed_scores"]) == 5
    assert 0.0 <= result["stability_score"] <= 100.0
