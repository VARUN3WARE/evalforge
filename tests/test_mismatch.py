from evalforge.mismatch import detect_confidence_accuracy_mismatch


def test_detect_confidence_accuracy_mismatch_binary_probs():
    """
    Flags highly confident wrong predictions like a responsible adult :)
    """
    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.05, 0.9, 0.95]

    result = detect_confidence_accuracy_mismatch(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        confidence_threshold=0.9,
    )

    assert result["total_samples"] == 4
    assert result["mismatch_count"] == 2
    assert result["mismatch_indices"] == [1, 3]
    assert result["mismatch_rate"] == 0.5


def test_detect_confidence_accuracy_mismatch_multiclass_probs():
    """
    Supports 2D probability arrays without emotional breakdown.
    """
    y_true = [0, 1, 2]
    y_pred = [0, 2, 2]
    y_prob = [
        [0.95, 0.03, 0.02],
        [0.02, 0.04, 0.94],
        [0.01, 0.03, 0.96],
    ]

    result = detect_confidence_accuracy_mismatch(y_true, y_pred, y_prob, confidence_threshold=0.9)
    assert result["mismatch_count"] == 1
    assert result["mismatch_indices"] == [1]
