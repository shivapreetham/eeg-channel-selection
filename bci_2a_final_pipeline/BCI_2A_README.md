# BCI Competition IV 2a Pipeline - README

## Overview
This pipeline adapts the PhysioNet motor imagery analysis methodology to the BCI Competition IV 2a dataset. The notebooks train and evaluate EEG-ARNN models and legacy baseline methods on the 4-class motor imagery task.

## Dataset Specifications

### BCI Competition IV 2a Dataset
- **Subjects**: 9 subjects (A01-A09)
- **Sessions**: 2 per subject
  - T (Training): ~288 trials
  - E (Evaluation): ~288 trials
- **Classes**: 4 motor imagery tasks
  - 769: Left hand
  - 770: Right hand
  - 771: Feet
  - 772: Tongue
- **Channels**: 22 EEG channels
  - Standard 10-20 system: Fz, C3, Cz, C4, Pz, and others
  - 3 EOG channels (excluded in analysis)
- **Sampling Rate**: 250 Hz
- **Trial Structure**:
  - t=0s: Fixation cross
  - t=2s: Cue onset (beep)
  - t=2-6s: Motor imagery (4 seconds)
  - t=6s: Short break
- **Epochs**: 0.5s to 4.5s relative to cue onset (4 seconds of MI)
- **Baseline**: -0.5s to 0s relative to cue onset

## Channel Selection Strategy

### Why k=[5, 8, 10, 12, 15]?

With only **22 EEG channels** (vs 64 in PhysioNet), we need smaller k values while maintaining comparable channel reduction percentages:

| Dataset | k values | Percentage Range |
|---------|----------|------------------|
| **PhysioNet** | [10, 15, 20, 25, 30] | 16% - 47% of 64 channels |
| **BCI 2a** | [5, 8, 10, 12, 15] | 23% - 68% of 22 channels |

**Rationale for each k:**

1. **k=5 (23% of channels)**
   - Minimal channel set
   - Tests extreme channel reduction
   - Useful for ultra-low-cost BCI systems
   - Expected accuracy drop: 10-20%

2. **k=8 (36% of channels)**
   - Moderate reduction
   - Likely includes primary motor cortex (C3, Cz, C4)
   - Good balance between cost and performance
   - Expected accuracy drop: 5-10%

3. **k=10 (45% of channels)**
   - Nearly half the channels
   - Should capture most relevant information
   - Comparable to k=20 in PhysioNet (31%)
   - Expected accuracy drop: 3-8%

4. **k=12 (55% of channels)**
   - Majority of channels
   - High information retention
   - Comparable to k=25 in PhysioNet (39%)
   - Expected accuracy drop: 2-5%

5. **k=15 (68% of channels)**
   - Conservative reduction
   - Removes only 7 channels (noisy/redundant)
   - Maximum performance with some reduction
   - Expected accuracy drop: 1-3%

### Channel Selection Methods

For each k value, three methods are evaluated:

1. **ES (Edge Selection)** - Baseline & Adaptive
   - Selects channels based on graph edge importance
   - Favors channels with strong inter-channel connectivity

2. **AS (Aggregation Selection)** - Baseline & Adaptive
   - Selects channels based on aggregated adjacency scores
   - Favors channels that aggregate information from neighbors

3. **GS (Gate Selection)** - Adaptive only
   - Selects channels based on learned gating values
   - Data-driven selection from adaptive gating mechanism

## Pipeline Structure

### Notebook 01: EEG-ARNN Models
[bci_2a_01_eeg_arnn_models.ipynb](bci_2a_01_eeg_arnn_models.ipynb)

**Models:**
- Baseline EEG-ARNN (pure CNN-GCN architecture)
- Adaptive Gating EEG-ARNN (input-dependent channel gating)

**Training:**
- 35 epochs, learning rate 0.001
- 3-fold cross-validation
- No early stopping (patience=999)
- Batch size: 64

**Outputs:**
- `bci_2a_baseline_results.csv` - Full 22-channel results
- `bci_2a_adaptive_results.csv` - Full 22-channel results
- `bci_2a_baseline_retrain_results.csv` - Channel selection results
- `bci_2a_adaptive_retrain_results.csv` - Channel selection results

### Notebook 02: Legacy Methods
[bci_2a_02_legacy_methods.ipynb](bci_2a_02_legacy_methods.ipynb)

**Models:**
1. FBCSP - Filter Bank Common Spatial Patterns + LDA
2. CNN-SAE - CNN with Spatial Attention
3. EEGNet - Compact convolutional network
4. ACS-SE-CNN - Attention + Squeeze-Excitation CNN
5. G-CARM - Graph-based CARM

**Training:**
- Same configuration as EEG-ARNN models
- 30 epochs, learning rate 0.002
- 3-fold cross-validation

**Outputs:**
- `bci_2a_legacy_{model}_results.csv` - Full 22-channel results
- `bci_2a_legacy_{model}_retrain_results.csv` - Channel selection results

### Notebook 03: Comparison
[bci_2a_03_comparison.ipynb](bci_2a_03_comparison.ipynb)

**Analyses:**
1. Full-channel performance (22 channels)
2. Channel selection performance (k=[5,8,10,12,15])
3. Accuracy drop analysis
4. Optimal k-value identification
5. Statistical comparisons (paired t-tests)
6. Visualizations (box plots, bar charts, radar charts)

**Outputs:**
- Comparison tables and statistics
- Publication-quality visualizations
- Summary CSV files

## Multi-Class Evaluation Metrics

Since BCI 2a is a **4-class problem** (vs 2-class in PhysioNet), metrics are adapted:

| Metric | Computation |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Precision** | Macro-averaged across 4 classes |
| **Recall** | Macro-averaged across 4 classes |
| **F1-Score** | Macro-averaged across 4 classes |
| **AUC-ROC** | One-vs-rest, macro-averaged |

**Note:** Binary-specific metrics (specificity, sensitivity) are removed.

## Key Differences from PhysioNet

| Aspect | PhysioNet | BCI 2a |
|--------|-----------|--------|
| **File Format** | EDF | GDF |
| **Subjects** | 109 (27 after cleaning) | 9 (all clean) |
| **Classes** | 2 (left/right fist) | 4 (left hand, right hand, feet, tongue) |
| **Channels** | 64 | 22 |
| **Sampling Rate** | 160 Hz | 250 Hz |
| **Runs** | 12 (R03-R14) | 2 (T, E) |
| **Epochs** | -1.0 to 5.0s | 0.5 to 4.5s |
| **k values** | [10, 15, 20, 25, 30] | [5, 8, 10, 12, 15] |
| **Metrics** | Binary + specificity | Multi-class (macro) |

## Running the Pipeline

### Prerequisites
```bash
pip install mne numpy pandas matplotlib seaborn scikit-learn torch tqdm scipy
```

### Data Directory Structure
```
eeg-channel-selection/
├── data/
│   └── BCI_2a/
│       ├── A01T.gdf
│       ├── A01E.gdf
│       ├── A02T.gdf
│       ├── A02E.gdf
│       └── ...
└── bci_2a_final_pipeline/
    ├── bci_2a_01_eeg_arnn_models.ipynb
    ├── bci_2a_02_legacy_methods.ipynb
    ├── bci_2a_03_comparison.ipynb
    └── results/
```

### Execution Order
1. Run **bci_2a_01_eeg_arnn_models.ipynb**
   - Trains EEG-ARNN models
   - Performs channel selection and retraining
   - ~2-4 hours on GPU

2. Run **bci_2a_02_legacy_methods.ipynb**
   - Trains 5 legacy baseline methods
   - Performs channel selection and retraining
   - ~4-6 hours on GPU

3. Run **bci_2a_03_comparison.ipynb**
   - Loads all results
   - Generates comparison statistics
   - Creates visualizations
   - ~5 minutes

## Expected Results

### Full-Channel Performance (22 channels)
Based on BCI 2a literature:
- **Chance level**: 25% (4 classes)
- **Good performance**: 70-80%
- **State-of-the-art**: 80-85%

### Channel Selection Performance
Expected accuracy drops by k:
- **k=5**: 10-20% drop (still above chance)
- **k=8**: 5-10% drop
- **k=10**: 3-8% drop
- **k=12**: 2-5% drop
- **k=15**: 1-3% drop

### Method Rankings
Expected ranking (best to worst):
1. Adaptive Gating EEG-ARNN
2. Baseline EEG-ARNN
3. G-CARM
4. CNN-SAE
5. ACS-SE-CNN
6. EEGNet
7. FBCSP

## Implementation Notes

### GDF File Loading
BCI 2a uses GDF format, which requires:
```python
raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose='ERROR')
```

### Event Codes
MNE automatically maps BCI 2a event codes:
- Original: 769, 770, 771, 772
- Internal: 7, 8, 9, 10 (depends on MNE version)
- **Solution**: Use `filter_classes()` to remap to 0, 1, 2, 3

### Channel Names
BCI 2a channels are named:
- EEG-Fz, EEG-0, EEG-1, ..., EEG-C3, EEG-Cz, EEG-C4, ..., EEG-Pz

The preprocessing pipeline automatically:
1. Picks EEG channels only (excludes EOG)
2. Sets standard 10-20 montage
3. Applies CAR (Common Average Reference)

### Multi-Class AUC-ROC
For 4-class classification:
```python
roc_auc_score(labels, probs, multi_class='ovr', average='macro')
```
- `probs` must be (n_samples, n_classes) probability matrix
- Uses one-vs-rest strategy
- Macro-averages across all classes

## Troubleshooting

### Issue: "No subjects found"
- Check data directory path in CONFIG
- Ensure GDF files are named correctly (A01T.gdf, A01E.gdf, etc.)
- Verify files are in the root of `data/BCI_2a/` (not subdirectories)

### Issue: "Channels mismatch"
- BCI 2a has 22 EEG + 3 EOG = 25 total channels
- Pipeline automatically picks only EEG channels
- Final channel count should be 22

### Issue: "Class imbalance"
- BCI 2a has balanced classes (~72 trials per class per session)
- Total: ~576 trials per subject
- Check data loading if imbalance occurs

### Issue: "Low accuracy (< 30%)"
- Check epoch time window (should be 0.5-4.5s)
- Verify event codes (769, 770, 771, 772)
- Ensure classes are remapped to 0, 1, 2, 3

## References

1. Tangermann et al. (2012). "Review of the BCI Competition IV." Frontiers in Neuroscience.
2. Ang et al. (2012). "Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b." Frontiers in Neuroscience.
3. Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces." Journal of Neural Engineering.

## Citation

If you use this pipeline, please cite:
```bibtex
@software{bci2a_eeg_arnn_pipeline,
  title={BCI Competition IV 2a - EEG-ARNN Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Last Updated**: November 2024
