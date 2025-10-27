# Architecture Variant Analysis Report
Generated: 2025-10-25 15:49:23

## Objective
Evaluate architectural improvements to EEG-ARNN to find optimal configuration.

## Variants Tested

- **baseline**: 3x (TFEM + 1-layer CARM)
- **multi2**: 3x (TFEM + 2-layer CARM) - 2 hops
- **multi3**: 3x (TFEM + 3-layer CARM) - 3 hops

## Results

**multi2**
- Accuracy: 0.6667 +/- 0.0000
- Time: 49.4s per fold

**multi3**
- Accuracy: 0.6667 +/- 0.0000
- Time: 65.2s per fold

**baseline**
- Accuracy: 0.6339 +/- 0.0732
- Time: 34.2s per fold

## Key Findings

1. **Best variant**: multi2 (0.6667)
2. **Baseline**: 0.6339
3. **Improvement**: +5.17%

## Recommendations

The **multi2** architecture shows significant improvement over baseline. Recommend using this configuration for final experiments.
