import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from evalforge.api import ModelAuditor

class TestBlindSpotsIntegration(unittest.TestCase):
    """
    Test that blind spot detection integrates smoothly with ModelAuditor.
    Because finding the model's "uh-oh" zones is better than pretending they don't exist :)
    """
    
    def setUp(self):
        # Create a simple synthetic dataset where the model will clearly fail in one region
        # Region 1: x < 0, target is 0
        # Region 2: x > 0, target is 1
        # Blind spot: x > 2, target is 0 (model will predict 1)
        
        np.random.seed(42)
        X = np.random.uniform(-5, 5, size=(200, 2))
        y = (X[:, 0] > 0).astype(int)
        
        # Introduce a blind spot in the top-right corner
        blind_mask = (X[:, 0] > 2) & (X[:, 1] > 2)
        y[blind_mask] = 0 
        
        self.df = pd.DataFrame(X, columns=["feat1", "feat2"])
        self.df["target"] = y
        
        self.model = LogisticRegression()
        self.model.fit(X, y)
        
    def test_auditor_blind_spots(self):
        auditor = ModelAuditor(self.model, target_col="target")
        
        # Run evaluation with blind spots enabled
        results = auditor.evaluate(self.df, run_blind_spots=True)
        
        # 1. Check health score data contains blind spots component
        self.assertIn("blind_spots", results["component_scores"])
        
        # 2. Check blind spots data exists on auditor
        self.assertIsNotNone(auditor.blind_spots_data_)
        self.assertIn("cluster_report", auditor.blind_spots_data_)
        
        # 3. Check report generation includes the blind spots section
        report = auditor.get_report()
        self.assertIn("Cluster-wise Blind Spots", report)
        self.assertIn("Blind Spots Detected", report)
        
    def test_auditor_no_blind_spots(self):
        auditor = ModelAuditor(self.model, target_col="target")
        
        # Run evaluation with blind spots DISABLED
        results = auditor.evaluate(self.df, run_blind_spots=False)
        
        # 1. Component score should be missing
        self.assertNotIn("blind_spots", results["component_scores"])
        
        # 2. Blind spots data should be None
        self.assertIsNone(auditor.blind_spots_data_)
        
        # 3. Report should not have the section
        report = auditor.get_report()
        self.assertNotIn("Cluster-wise Blind Spots", report)

if __name__ == "__main__":
    unittest.main()
