import pytest
import pandas as pd
import numpy as np
import sys # Don't forget to import sys, otherwise the CLI tests get all sulky :)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from evalforge.api import ModelAuditor
from evalforge.stability import compute_stability_from_scores
from evalforge.report_card import generate_report_card
import argparse # For mocking CLI args

@pytest.fixture
def mock_pipeline():
    """Generates a small dataset and model to abuse during API tests."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    df["target"] = y
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, df

def test_model_auditor_stability_integration(mock_pipeline):
    """
    Test that ModelAuditor correctly integrates stability scores.
    Because a stable model is a happy model, and a happy developer :)
    """
    model, df = mock_pipeline
    auditor = ModelAuditor(model)
    
    # Test with stability scores
    test_stability_scores = [0.8, 0.85, 0.79, 0.82, 0.83]
    results = auditor.evaluate(df_test=df, stability_scores=test_stability_scores)
    
    assert "health_score" in results
    assert auditor.stability_data_ is not None
    assert "stability_score" in auditor.stability_data_
    assert results["component_scores"]["stability"] == pytest.approx(auditor.stability_data_["stability_score"])

    # Test without stability scores
    auditor_no_stability = ModelAuditor(model)
    results_no_stability = auditor_no_stability.evaluate(df_test=df)
    assert auditor_no_stability.stability_data_ is None
    assert "stability" not in results_no_stability["component_scores"]

def test_report_card_stability_display(mock_pipeline):
    """
    Test that the report card displays stability information correctly.
    Because what good is data if you can't show it off? ;)
    """
    model, df = mock_pipeline
    auditor = ModelAuditor(model)
    
    test_stability_scores = [0.8, 0.85, 0.79, 0.82, 0.83]
    auditor.evaluate(df_test=df, stability_scores=test_stability_scores)
    
    report = auditor.get_report()
    
    assert "Stability Score" in report
    assert f"{auditor.stability_data_['stability_score']:.1f}%" in report
    
    # Test report when no stability data is provided
    auditor_no_stability = ModelAuditor(model)
    auditor_no_stability.evaluate(df_test=df)
    report_no_stability = auditor_no_stability.get_report()
    assert "Stability Score" not in report_no_stability

def test_cli_stability_scores_argument(mock_pipeline, capsys):
    """
    Test the CLI's ability to process stability scores argument.
    Because even the command line needs to understand emotional intelligence :)
    """
    # Mock argparse.ArgumentParser and sys.argv for testing
    from evalforge import cli
    
    # Mock objects for cli.main
    cli.load_pkl = lambda x: mock_pipeline[0]
    cli.pd.read_csv = lambda x: mock_pipeline[1]

    # Test with valid stability scores
    sys.argv = [
        "evalforge", "analyze", 
        "--model", "dummy_model.pkl", 
        "--data", "dummy_data.csv", 
        "--target", "target",
        "--stability-scores", "0.9,0.85,0.92,0.88"
    ]
    cli.main()
    captured = capsys.readouterr()
    assert "Calculating Health Score..." in captured.out
    assert "Stability Score" in captured.out
    # Removed: assert "Passed" in captured.out # This only appears if --fail-under is used

    # Test with invalid stability scores
    sys.argv = [
        "evalforge", "analyze", 
        "--model", "dummy_model.pkl", 
        "--data", "dummy_data.csv", 
        "--target", "target",
        "--stability-scores", "0.9,abc,0.92"
    ]
    cli.main()
    captured_invalid = capsys.readouterr()
    assert "Invalid format for --stability-scores." in captured_invalid.out
