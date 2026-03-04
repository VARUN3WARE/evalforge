import os
import shutil
import pandas as pd
from evalforge.visualize import plot_fragility_drop, plot_drift_histogram

def test_plot_fragility_drop():
    """
    Test generating the fragility drop plot without crashing matplotlib.
    If it crashes, graphical libraries are still painful :)
    """
    test_dir = "test_reports"
    os.makedirs(test_dir, exist_ok=True)
    
    baseline = 0.95
    drops = {"gaussian_noise": 0.05, "feature_masking": 0.10, "feature_scaling": 0.02}
    
    path = plot_fragility_drop(baseline, drops, output_dir=test_dir)
    
    assert os.path.exists(path)
    assert path.endswith("fragility_drop.png")
    
    # Cleanup
    shutil.rmtree(test_dir)

def test_plot_drift_histogram():
    """
    Test generating the drift feature histogram without complaints.
    """
    test_dir = "test_reports_drift"
    os.makedirs(test_dir, exist_ok=True)
    
    train = pd.DataFrame({"A": [1, 2, 3, 4, 1, 2, 3]})
    test = pd.DataFrame({"A": [5, 6, 7, 8, 5, 6, 7]})
    
    path = plot_drift_histogram(train, test, "A", output_dir=test_dir)
    
    assert os.path.exists(path)
    assert path.endswith("drift_A.png")
    
    # Cleanup
    shutil.rmtree(test_dir)
