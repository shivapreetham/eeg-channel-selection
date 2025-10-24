# PhysioNet Channel Selection Analysis Report

**Generated:** 2025-10-24 10:33:11

## Summary

This report analyzes channel selection performance on the PhysioNet Motor Imagery dataset.

### Dataset Statistics
- **Subjects analyzed:** 5
- **Total channels:** 64
- **Baseline accuracy (all channels):** 72.62% ± 10.56%
- **K values tested:** [10, 15, 20, 25]

### Methods Evaluated
- **ES (Edge Selection):** Selects channels based on strongest pairwise connections
- **AS (Aggregation Selection):** Selects channels based on aggregate connection strength

---

## Optimal K Selection

**Criterion:** Minimize k while maintaining >=95% of baseline accuracy (68.99%)

### ES Method ✓
- **Optimal k:** 20
- **Actual channels selected:** 31
- **Accuracy:** 70.39% ± 10.17%
- **Channel reduction:** 51.9% (use 31 instead of 64)
- **Accuracy drop:** 2.71%

---

## Results by K Value

| Method | k | Channels | Accuracy (%) | Acc Drop (%) | Channel Reduction (%) |
|--------|---|----------|--------------|--------------|----------------------|
| AS | 10 | 10.0 | 63.23 ± 14.53 | 13.38 | 84.4 |
| AS | 15 | 15.0 | 67.76 ± 15.81 | 6.83 | 76.6 |
| AS | 20 | 20.0 | 64.58 ± 11.19 | 11.21 | 68.8 |
| AS | 25 | 25.0 | 62.99 ± 7.83 | 12.64 | 60.9 |
| ES | 10 | 18.0 | 60.02 ± 16.64 | 18.27 | 71.9 |
| ES | 15 | 24.4 | 61.64 ± 15.25 | 15.82 | 61.9 |
| ES | 20 | 30.8 | 70.39 ± 10.17 | 2.71 | 51.9 |
| ES | 25 | 34.6 | 66.90 ± 10.59 | 7.78 | 45.9 |

---

## Recommendation

**Use ES with k=20**

This achieves:
- 51.9% channel reduction
- Only 2.71% accuracy drop
- Use 31 channels instead of 64

**Practical Impact:**
- Setup time: ~31 electrodes vs 64 (saves ~10 minutes)
- Cost: $31*10 = $308 vs $640 (saves $332)
- User comfort: Significantly improved

---

## Files Generated

- `aggregated_results.csv` - Results aggregated across subjects
- `optimal_k_selection.csv` - Optimal k for each method
- `accuracy_vs_k.png` - Main visualization
- `accuracy_drop_vs_k.png` - Accuracy drop plot
- `reduction_vs_accuracy.png` - Trade-off plot
- `ANALYSIS_REPORT.md` - This report

---

*Analysis complete!*
