import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from evalforge.api import ModelAuditor

class TestStabilityIntegration(unittest.TestCase):
    """
    Test that stability scoring integrates properly with the health score system.
    Because consistency is just accuracy that doesn't change its mind :)
    """
    
    def setUp(self):
        # Dummy data for ModelAuditor initialization
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        self.df = pd.DataFrame(X, columns=["f1", "f2"])
        self.df["target"] = y
        self.model = LogisticRegression().fit(X, y)
        
    def test_stability_weighting(self):
        auditor = ModelAuditor(self.model, target_col="target")
        
        # 1. Perfectly stable scores (all 0.9 accuracy)
        stable_scores = [0.9, 0.9, 0.9]
        results_stable = auditor.evaluate(self.df, stability_scores=stable_scores)
        
        # 2. Highly unstable scores
        unstable_scores = [0.1, 0.9, 0.5]
        results_unstable = auditor.evaluate(self.df, stability_scores=unstable_scores)
        
        # Check component scores
        self.assertIn("stability", results_stable["component_scores"])
        self.assertIn("stability", results_unstable["component_scores"])
        
        # Stable should have higher stability component than unstable
        self.assertGreater(
            results_stable["component_scores"]["stability"], 
            results_unstable["component_scores"]["stability"]
        )
        
        # 3. Report Card check
        report = auditor.get_report()
        self.assertIn("Stability Analysis", report)
        self.assertIn("Stability Score", report)

if __name__ == "__main__":
    unittest.main()
