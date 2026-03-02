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
