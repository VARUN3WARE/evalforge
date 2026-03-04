import subprocess
import sys

def test_cli_help():
    """
    Test that the CLI boots up and shows help without exploding.
    If this fails, the entry point is broken :)
    """
    # Simply run python -m evalforge.cli -h
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.cli", "-h"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "EvalForge: Model Health & Evaluation Intelligence Engine." in result.stdout
    assert "analyze" in result.stdout
