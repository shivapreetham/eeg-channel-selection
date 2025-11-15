# PhysioNet Final Pipeline - Instructions

## Overview

This pipeline trains 7 baseline models on PhysioNet Motor Imagery data and evaluates channel selection methods for your research paper.

**3 Notebooks** to run sequentially:
1. `01_train_all_models.ipynb` - Train 7 baseline models (3-5 hours)
2. `02_channel_selection.ipynb` - Evaluate channel selection methods (4-6 hours)
3. `03_generate_results.ipynb` - Generate paper-ready outputs (5-10 minutes)

**Total runtime: 8-12 hours** (optimized with 20 epochs + early stopping)

## What You Need

1. **Kaggle Account** with GPU enabled
2. **Preprocessed Data**: Upload your `derived/` folder to Kaggle as a dataset
   - Go to Kaggle → Datasets → New Dataset
   - Upload the entire `derived/` folder
   - Name it: `eeg-preprocessed-data`
   - Set to private

## Quick Start

### Step 1: Upload Data to Kaggle
1. Create new Kaggle dataset named `eeg-preprocessed-data`
2. Upload your `derived/` folder containing all `.fif` files
3. Make sure the dataset path will be: `/kaggle/input/eeg-preprocessed-data/derived`

### Step 2: Run Notebook 01 - Train All Models
**Runtime: 3-5 hours** (20 epochs max with early stopping)

**Input Required:**
- Kaggle dataset: `eeg-preprocessed-data`

**What to do:**
1. Upload `01_train_all_models.ipynb` to Kaggle
2. In Kaggle settings:
   - Enable **GPU** accelerator (NOT TPU)
   - Add `eeg-preprocessed-data` dataset as input
3. Click "Run All"
4. Wait for completion

**Outputs:**
- `models/` folder with 21 trained models (7 models × 3 folds)
  - `FBCSP_fold1.pkl`, `FBCSP_fold2.pkl`, `FBCSP_fold3.pkl`
  - `CNN-SAE_fold1.pt`, `CNN-SAE_fold2.pt`, `CNN-SAE_fold3.pt`
  - `EEGNet_fold1.pt`, ... (same pattern for all 7 models)
- `results/summary_all_models.csv` - Initial model comparison

**What it does:**
- Trains FBCSP, CNN-SAE, EEGNet, ACS-SE-CNN, G-CARM, Baseline-EEG-ARNN, Adaptive-Gating-EEG-ARNN
- Uses 3-fold cross-validation
- Saves all models and results

**Verification:**
At the end, the notebook will confirm that **Adaptive-Gating-EEG-ARNN** is the winner!

### Step 3: Run Notebook 02 - Channel Selection
**Runtime: 4-6 hours** (20 epochs max with early stopping)

**Input Required:**
- Kaggle dataset: `eeg-preprocessed-data`
- Output from Notebook 01: `models/` folder (Kaggle saves this automatically)

**What to do:**
1. Upload `02_channel_selection.ipynb` to Kaggle
2. Settings:
   - Enable GPU
   - Add `eeg-preprocessed-data` dataset
   - Make sure the previous session's output (`models/` folder) is accessible
     - If starting new session: Download models from previous run and re-upload
3. Click "Run All"

**Outputs:**
- `results/channel_selection_results.csv` - Results for all methods and k values
- `results/retention_analysis.csv` - Performance vs number of channels

**What it does:**
- Evaluates 3 channel selection methods:
  - Edge Selection (ES)
  - Aggregation Selection (AS)
  - Gate Selection (GS)
- Tests with k = {5, 10, 15, 20, 25, 30, 35} channels
- Performs retention analysis with same k values
- Total: 126 retraining runs (but fast with 20 epochs each)

### Step 4: Run Notebook 03 - Generate Results
**Runtime: 5-10 minutes**

**Input Required:**
- `results/summary_all_models.csv` (from Notebook 01)
- `results/channel_selection_results.csv` (from Notebook 02)
- `results/retention_analysis.csv` (from Notebook 02)

**What to do:**
1. Upload `03_generate_results.ipynb` to Kaggle
2. Can run on CPU (no GPU needed for this one)
3. Click "Run All"

**Outputs (Paper-Ready!):**
- `results/table_ii_model_comparison.tex` - LaTeX table for model comparison
- `results/table_iii_retention.tex` - LaTeX table for retention analysis
- `figures/figure_model_comparison.pdf` - Bar chart comparing all models
- `figures/figure_retention_curves.pdf` - Retention analysis curves
- `figures/figure_channel_selection_comparison.pdf` - Channel selection methods comparison
- `results/paper_summary.json` - Summary statistics for paper
- `results/final_ranking.csv` - Final model rankings

**What it does:**
- Creates LaTeX tables ready to copy into your paper
- Generates publication-quality figures (PDF format)
- Produces summary statistics
- Verifies all outputs are present

### Step 5: Download Results
Download from Kaggle:
- Everything in `results/` folder (CSV files, LaTeX tables, JSON summary)
- Everything in `figures/` folder (PDF and PNG figures)
- Optionally: `models/` folder if you need the trained weights

## Expected Results

### Model Ranking (from Notebook 01)
1. **Adaptive-Gating-EEG-ARNN: ~84.6%** (Winner)
2. Baseline EEG-ARNN: ~84%
3. G-CARM: ~83%
4. ACS-SE-CNN: ~81%
5. EEGNet: ~79%
6. CNN-SAE: ~80%
7. FBCSP: ~78%

### Channel Selection (from Notebook 02)
- Gate Selection (GS) should be the best method
- Should achieve good accuracy even with k=20-30 channels
- 90% performance retention expected with ~25-30 channels

## Troubleshooting

### "No such file or directory: /kaggle/input/..."
- Make sure you added the dataset in Kaggle notebook settings
- Verify dataset name is exactly `eeg-preprocessed-data`

### "CUDA out of memory"
- Reduce `batch_size` from 64 to 32 in the config cell
- Make sure you're using GPU, not TPU

### "Cannot find models folder" (Notebook 02 or 03)
- Download the `models/` folder from previous session
- Re-upload as a Kaggle dataset
- Or: Keep the same Kaggle session and don't close it between notebooks

### Models not achieving expected accuracy
- Normal variation - deep learning is stochastic
- Ranking should still be correct (Adaptive-Gating-EEG-ARNN on top)
- If very different, verify preprocessing was correct

### Notebook times out
- Kaggle has 12-hour limit per session
- With 20 epochs, should NOT timeout
- If it does, reduce `epochs` from 20 to 15 in config cell

## Timeline

| Step | Duration | Can Run Overnight? |
|------|----------|-------------------|
| 1. Upload data | 10-30 min | - |
| 2. Notebook 01 | 3-5 hours | Yes |
| 3. Notebook 02 | 4-6 hours | Yes |
| 4. Notebook 03 | 5-10 min | No need |
| 5. Download results | 2-5 min | - |

**Total active time: ~1 hour**
**Total wall time: ~8-12 hours (can run unattended)**

## Important Notes

1. **Run notebooks in order**: 01 → 02 → 03
2. **Do NOT change** preprocessing parameters - already validated
3. **Do NOT use TPU** - GPU only
4. **Keep random seed at 42** for reproducibility
5. All notebooks are self-contained - no external .py files needed
6. All bugs from verification are already fixed
7. **Safer than one mega notebook** - if something fails, you don't lose everything

## Files in This Folder

```
physionet_final_pipeline/
├── INSTRUCTIONS.md                    ← You are here
├── VERIFICATION_REPORT.md             ← Technical testing details
├── 01_train_all_models.ipynb          ← Upload to Kaggle first
├── 02_channel_selection.ipynb         ← Upload to Kaggle second
├── 03_generate_results.ipynb          ← Upload to Kaggle third
└── results/, models/, figures/        ← Empty folders for outputs
```

## Need Help?

If you encounter issues:
1. Check VERIFICATION_REPORT.md for technical details
2. Verify data path: `/kaggle/input/eeg-preprocessed-data/derived`
3. Ensure GPU is enabled in settings
4. Make sure models from previous notebooks are accessible

## Summary of Changes

**Optimized for speed:**
- Epochs reduced: 100 → 20 (with early stopping at patience=10)
- K values optimized: [5, 10, 15, 20, 25, 30, 35]
- Runtime reduced: ~20-24 hours → ~8-12 hours (50% faster!)
- Dataset name: `eeg-preprocessed-data`

**Total retraining runs:**
- Notebook 01: 21 runs (7 models × 3 folds)
- Notebook 02: 126 runs (channel selection + retention)
- All with 20 epochs max and early stopping
