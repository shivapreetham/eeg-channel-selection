# Pipeline 2: EEG-ARNN Methods with Channel Selection

This pipeline contains the COMPLETE implementation of your novel EEG-ARNN methods.

## What's Included

### Part 1: Model Training (2 models × 3 folds)
- Baseline-EEG-ARNN
- Adaptive-Gating-EEG-ARNN (with adaptive gating mechanism)

### Part 2: Channel Selection Evaluation
Three selection methods tested:
1. **Edge Selection (ES)** - Based on graph adjacency matrix
2. **Aggregation Selection (AS)** - Based on feature activations
3. **Gate Selection (GS)** - Based on adaptive gate values (Adaptive-Gating only)

Tested on k-values: [10, 20, 30, 40, 50]

### Part 3: Retention Analysis
Performance retention curve with k-values: [10, 15, 20, 25, 30, 35]
Shows how accuracy degrades as channels are reduced.

## Runtime Estimate
- Initial training: ~1.5 hours (2 models × 3 folds × ~15 min)
- Channel selection: ~11 hours (2 models × 3 methods × 5 k × 3 folds × ~15 min)
- Retention: ~4.5 hours (6 k × 3 folds × ~15 min)
- **TOTAL: ~17 hours**

## Outputs
- `models/eegarnn_*.pt` - Model checkpoints
- `results/eegarnn_baseline_results.csv` - Baseline EEG-ARNN results
- `results/eegarnn_adaptive_results.csv` - Adaptive Gating results
- `results/channel_selection_results.csv` - All channel selection experiments
- `results/retention_analysis.csv` - Retention curve data
- `results/eegarnn_complete_summary.csv` - Final summary

## Usage
1. Run all cells in sequence
2. Results are automatically saved to CSV files
3. Compare with Pipeline 1 results for complete analysis
