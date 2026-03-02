# EvalForge

An Open-Source Model Health & Evaluation Intelligence Engine

EvalForge is a pre-deployment reliability engine for machine learning models.  
It evaluates robustness, calibration, drift, stability, and hidden failure modes, summarized into a unified Model Health Score (0-100).

---

## Why EvalForge?

Accuracy alone is not enough. A model with 94% accuracy may still be poorly calibrated, collapse under slight noise, drift in production, show high variance, or produce highly confident wrong predictions. EvalForge detects these risks before deployment.

---

## Core Features (v0.1)

- Model Health Score (0-100)
- Bootstrap Confidence Intervals
- Confidence-Accuracy Mismatch Detection
- Adversarial Fragility Score
- Drift Detection (KS test)
- Seed Stability Testing
- Automated Evaluation Report Card

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Developer Guide (For Phase 3 & Beyond)

If you are picking up development from here, follow these steps to keep everything building correctly:

### 1. Environment Setup

We strongly recommend using `uv` for blistering fast environments:

```bash
# Create the virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install the dependencies
uv pip install -r requirements.txt
```

### 2. Running Tests

We use `pytest`. If tests are failing, please grab a coffee and fix them before opening a PR :)

```bash
# Run all tests
pytest tests/
```

### 3. Adding New Features

Code should go in the `evalforge/` directory, and your tests should go in the `tests/` directory.

- Please ensure your docstrings have a tiny, simple 1-line joke :)
- Stick to the SOLID principles.
- Don't use emojis in the main documentation.
