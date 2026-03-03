import numpy as np
from evalforge.bootstrap import compute_bootstrap_ci

def test_compute_bootstrap_ci():
    """
    Test that the bootstrap CI function doesn't crash and returns sane bounds.
    If it fails, it means my statistics degree is useless :(
    """
    # Create some dummy data
    np.random.seed(42)  # Because we respect reproducible chaos
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.copy(y_true)
    y_pred[:10] = 1 - y_pred[:10]  # Flip 10% to make the accuracy 90%
    y_prob = np.random.rand(100)
    
    results = compute_bootstrap_ci(y_true, y_pred, y_prob, n_iterations=100)
    
    assert "accuracy_ci" in results
    assert "f1_ci" in results
    assert "auc_ci" in results
    
    lower_acc, upper_acc = results["accuracy_ci"]
    
    # 90% is within the boundary bounds
    assert lower_acc <= 0.9 <= upper_acc
    
    # Check that bounds make sense (lower <= upper)
    assert lower_acc <= upper_acc
    
def test_compute_bootstrap_no_prob():
    """
    Test what happens when the user is too lazy to provide probabilities.
    """
    y_true = [0, 1, 0, 1, 0, 1] * 10
    y_pred = [0, 1, 0, 0, 0, 1] * 10
    
    results = compute_bootstrap_ci(y_true, y_pred, n_iterations=50)
    
    assert "accuracy_ci" in results
    assert "auc_ci" not in results
