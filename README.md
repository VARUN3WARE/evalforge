# EvalForge

An Open-Source Model Health & Evaluation Intelligence Engine

EvalForge is a pre-deployment reliability engine for machine learning models. It evaluates robustness, calibration, drift, stability, and hidden failure modes, summarized into a unified Model Health Score (0-100).

---

## Current Capabilities

- Model Health Score (0-100): Weighted analysis across accuracy, calibration, fragility, drift, stability, and blind spots.
- Python API (ModelAuditor): Clean, object-oriented interface for orchestrating evaluations programmatically.
- CLI Analysis: Robust command-line tool with flags for stability and cluster-wise blind spot mapping.
- Diagnostics Layer:
  - Confidence-Accuracy Mismatch Detection.
  - Adversarial Fragility Scoring.
  - Distribution Drift Detection (KS test).
  - Seed Stability Analysis (Mean/Std across runs).
  - Cluster-wise Blind Spot Mapping (K-Means).
- Reporting: Structured markdown report cards and shareable HTML exports with visual evidence.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Python API Usage

```python
from evalforge.api import ModelAuditor
import pandas as pd
import pickle

# Load your model and data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
df_test = pd.read_csv("test_data.csv")

# Initialize and evaluate
auditor = ModelAuditor(model, target_col="target")
health_score = auditor.evaluate(df_test, run_blind_spots=True)

# Generate and export report
report = auditor.get_report()
auditor.export_report("reports/my_report.html")
```

### CLI Usage

```bash
evalforge analyze \
  --model model.pkl \
  --data test.csv \
  --target "target" \
  --stability "0.91,0.89,0.92" \
  --blind-spots \
  --export-html
```

---

## Next Steps

1. Regression Support: Implementation of RMSE, MAE, and R-squared metrics in the metrics layer and residual-based mismatch detection.
2. Fairness & Bias Evaluation: Calculation of metric disparity (Demographic Parity Ratio, Equal Opportunity Difference) and integrated bias penalties.
3. CI/CD Integration: Finalizing GitHub Action templates to enforce health score thresholds during deployment pipelines.
4. Technical Polish: Addition of full type hinting across the codebase and integration of the logging module for cleaner UI output.

---

## Developer Guide

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests

```bash
python3 -m pytest tests/
```

- Please ensure your docstrings have a tiny, simple 1-line joke :)
- Stick to the SOLID principles.
- Do not use emojis in the documentation.
