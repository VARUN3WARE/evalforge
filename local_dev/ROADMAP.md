# EvalForge Roadmap: Feature Enhancements & Improvements

## Background & Motivation
EvalForge currently provides a solid foundation for model evaluation with its "Model Health Score" approach. However, several core modules (`stability.py`, `blind_spots.py`) are not yet integrated into the main `ModelAuditor` and CLI workflows. Additionally, existing modules like `fairness.py` and `drift.py` could benefit from more robust statistical methods.

## Scope & Impact
The goal is to transition EvalForge from a collection of diagnostic scripts into a fully integrated, production-ready evaluation intelligence engine. This will improve model reliability assessments and provide deeper insights for ML engineers.

---

## Proposed Solution & Implementation Plan

### Phase 1: Core Integration & Stability
- **Integrate Stability Module**: Update `ModelAuditor` to support seed stability testing. This will require a way for users to provide a training function or multiple model checkpoints.
- **Health Score Updates**: Ensure `stability_score` is correctly passed and weighted in `compute_health_score`.
- **CLI Support**: Add `--stability` flag to the `analyze` command.

### Phase 2: Deep Diagnostics (Blind Spots)
- **Integrate Blind Spot Detection**: Add `map_blind_spots` to the `ModelAuditor.evaluate` pipeline.
- **Reporting**: Include a "Blind Spots" section in the Markdown and HTML reports to highlight feature regions with low performance.
- **Visualization**: Create a visualization for cluster-wise performance in `visualize.py`.

### Phase 3: Advanced Fairness & Drift
- **Enhanced Fairness**: Add "Equalized Odds" and "Equality of Opportunity" to `fairness.py` for more comprehensive bias checking.
- **Multiple Drift Methods**: Introduce PSI (Population Stability Index) and Jensen-Shannon divergence to `drift.py` alongside the current KS test.
- **Automated Thresholding**: Improve how drift thresholds are set (e.g., Bonferroni correction for multiple hypothesis testing).

### Phase 4: Reporting & UX
- **Interactive HTML**: Upgrade `exports.py` to use a more modern, interactive reporting template (potentially with Plotly or D3.js).
- **Bootstrap Integration**: Ensure all metrics in the report include confidence intervals using the `bootstrap.py` module.
- **API Refinement**: Simplify the `ModelAuditor` interface to make it even more intuitive for Jupyter users.

---

## Verification & Testing
- **Unit Tests**: Add tests for new integration points in `test_api.py` and `test_cli.py`.
- **Regression Testing**: Ensure existing metrics and health scores remain consistent.
- **E2E Demo**: Update `demo.py` to showcase the new features (Blind Spots, Stability, etc.).

## Migration & Rollback
- Maintain backward compatibility for `ModelAuditor` by keeping new features optional (e.g., `run_stability=False` by default).
- Use feature flags in the CLI to avoid breaking existing CI/CD pipelines.
