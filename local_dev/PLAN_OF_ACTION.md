# Plan of Action: EvalForge Phase 1 Implementation

This document outlines the immediate next steps to implement Phase 1 of the roadmap, focusing on core integration and stability testing.

## Task 1: Stability Testing Integration

- **File**: `evalforge/api.py`
- **Action**: Add a `stability_scores` parameter to `ModelAuditor.evaluate` or a way to provide multiple checkpoints.
- **Action**: Call `compute_stability_from_scores` from `evalforge.stability`.
- **Action**: Pass the `stability_score` to `compute_health_score`.

## Task 2: Blind Spot Detection Integration

- **File**: `evalforge/api.py`
- **Action**: Integrate `map_blind_spots` into the `evaluate` method.
- **Action**: Store `blind_spot_data` in the `ModelAuditor` instance.

## Task 3: Health Score Weighting

- **File**: `evalforge/health_score.py`
- **Action**: Ensure `stability` and other new metrics are properly weighted (as already drafted in `compute_health_score` but not fully utilized).

## Task 4: CLI Flags

- **File**: `evalforge/cli.py`
- **Action**: Add `--stability` (expecting a list of scores or a training function).
- **Action**: Add `--blind-spots` to trigger the cluster-wise analysis.

## Task 5: Reporting Layer

- **File**: `evalforge/report_card.py`
- **Action**: Add sections for "Stability Analysis" and "Cluster-wise Blind Spots".
- **File**: `evalforge/exports.py`
- **Action**: Update HTML template to render new sections.

---

## Verification Strategy

1. **Unit Tests**:
   - Create `tests/test_stability_integration.py`.
   - Create `tests/test_blind_spots_integration.py`.
2. **Integration Test**:
   - Run the updated `ModelAuditor` on synthetic data in `demo.py`.
3. **CLI Test**:
   - Run `evalforge analyze` with the new flags and verify the output markdown.

---

**Next Step**: Start implementation of Task 1 (Stability Integration).
:)
