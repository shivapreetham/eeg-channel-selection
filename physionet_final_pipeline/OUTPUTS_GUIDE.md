# Complete Outputs Guide - For Later Comparison

## Pipeline 1 Outputs

### 1. baseline_methods_results.csv
```csv
model,fold,accuracy
FBCSP,1,0.6142
FBCSP,2,0.6108
FBCSP,3,0.6178
CNN-SAE,1,0.6523
...
```
**Use for:** Detailed per-fold results for each baseline method

### 2. baseline_methods_summary.csv
```csv
rank,model,mean_accuracy,std_accuracy
1,EEGNet,0.8402,0.0030
2,CNN-SAE,0.6973,0.1272
3,G-CARM,0.5078,0.0115
4,ACS-SE-CNN,0.5459,0.0324
5,FBCSP,0.6143,0.0053
```
**Use for:** Quick rankings and mean accuracies

### 3. baseline_retention_analysis.csv
```csv
model,k,mean_accuracy,std_accuracy,fold_accuracies
EEGNet,10,0.7234,0.0123,[0.7345,0.7123,0.7234]
EEGNet,15,0.7456,0.0098,[0.7567,0.7345,0.7456]
EEGNet,20,0.7678,0.0087,[0.7789,0.7567,0.7678]
...
```
**Use for:** Variance-based retention curve (comparison baseline)

---

## Pipeline 2 Outputs

### 1. eegarnn_baseline_eeg_arnn_results.csv
```csv
fold,accuracy,model
1,0.5260,Baseline-EEG-ARNN
2,0.5344,Baseline-EEG-ARNN
```
**Use for:** Baseline-EEG-ARNN performance (no gating)

### 2. eegarnn_adaptive_gating_eeg_arnn_results.csv
```csv
fold,accuracy,model
1,0.7644,Adaptive-Gating-EEG-ARNN
2,0.7905,Adaptive-Gating-EEG-ARNN
```
**Use for:** Adaptive-Gating-EEG-ARNN performance (WITH gating)

### 3. channel_selection_results.csv
```csv
model,method,k,mean_accuracy,std_accuracy,fold_accuracies
Baseline-EEG-ARNN,edge,10,0.5234,0.0123,[0.5345,0.5123]
Baseline-EEG-ARNN,edge,20,0.5456,0.0098,[0.5567,0.5345]
Baseline-EEG-ARNN,aggregation,10,0.5345,0.0111,[0.5456,0.5234]
...
Adaptive-Gating-EEG-ARNN,edge,10,0.6234,0.0145,[0.6345,0.6123]
Adaptive-Gating-EEG-ARNN,aggregation,10,0.6456,0.0132,[0.6567,0.6345]
Adaptive-Gating-EEG-ARNN,gate,10,0.7234,0.0098,[0.7345,0.7123]  ← YOUR METHOD!
Adaptive-Gating-EEG-ARNN,gate,20,0.7456,0.0087,[0.7567,0.7345]
...
```
**Use for:**
- Compare Edge vs Aggregation vs Gate selection
- Show Gate selection is best
- Show performance at different k-values

### 4. retention_analysis.csv
```csv
k,mean_accuracy,std_accuracy,fold_accuracies
10,0.6234,0.0123,[0.6345,0.6123]
15,0.6789,0.0098,[0.6890,0.6688]
20,0.7123,0.0087,[0.7234,0.7012]
25,0.7456,0.0076,[0.7567,0.7345]
30,0.7678,0.0065,[0.7789,0.7567]
35,0.7834,0.0054,[0.7945,0.7723]
```
**Use for:** Gate-based retention curve (YOUR METHOD)

---

## What You Can Compare Later

### Comparison 1: All 7 Models
Combine:
- `baseline_methods_summary.csv` (5 models)
- Calculate mean/std from Pipeline 2 EEG-ARNN results (2 models)

**Result:** Table showing all 7 models ranked

### Comparison 2: Retention Curves
Compare:
- `baseline_retention_analysis.csv` (EEGNet, variance-based)
- `retention_analysis.csv` (Adaptive-Gating, gate-based)

**Result:** Plot showing adaptive gating retains performance better

### Comparison 3: Channel Selection Methods
From `channel_selection_results.csv`:
- Filter for Adaptive-Gating-EEG-ARNN
- Compare edge vs aggregation vs gate
- Show gate is best at each k-value

**Result:** Table/plot showing gate selection superiority

### Comparison 4: Impact of Adaptive Gating
Compare:
- Baseline-EEG-ARNN (no gating)
- Adaptive-Gating-EEG-ARNN (with gating)

**Result:** Show improvement from adding adaptive gating

---

## Summary

✅ **Pipeline 1 gives you:** 5 baseline methods + variance retention
✅ **Pipeline 2 gives you:** 2 EEG-ARNN methods + channel selection + gate retention

**All metrics saved separately** - you can combine and compare however you want later!

No built-in comparison in the pipelines - completely separate and modular! 👍
