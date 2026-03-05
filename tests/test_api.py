import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from evalforge.api import ModelAuditor

@pytest.fixture
def mock_pipeline():
    """Generates a small dataset and model to abuse during API tests."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    df["target"] = y
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, df

def test_model_auditor_init(mock_pipeline):
    """
    Test initialization logic, specifically the 'model must predict' rule.
    """
    model, _ = mock_pipeline
    
    # Happy path
    auditor = ModelAuditor(model)
    assert auditor.model is not None
    
    # Sad path
    class BadModel:
        pass
        
    with pytest.raises(TypeError):
        ModelAuditor(BadModel())

def test_model_auditor_evaluate(mock_pipeline):
    """
    Test that the .evaluate() pipeline runs seamlessly.
    If this breaks, Jupyter notebook users will cry :)
    """
    model, df = mock_pipeline
    auditor = ModelAuditor(model)
    
    # Evaluate with train and test
    results = auditor.evaluate(df_test=df, df_train=df, run_fragility=True)
    
    assert "health_score" in results
    assert auditor.health_score_data_ is not None
    assert auditor.mismatch_data_ is not None

def test_model_auditor_report(mock_pipeline):
    """
    Test that generating the report card from the API works.
    """
    model, df = mock_pipeline
    auditor = ModelAuditor(model)
    
    # Fails if you try to get a report before evaluating
    with pytest.raises(RuntimeError):
        auditor.get_report()
        
    # Works after evaluation
    auditor.evaluate(df, run_fragility=False)
    report = auditor.get_report()
    
    assert len(report) > 100
    assert "Model Health Score" in report
