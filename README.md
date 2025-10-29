# EEG Channel Selection for BCI (BCI Competition IV 2a)

This repository implements an end-to-end workflow for reducing the number of EEG channels required for motor imagery Brain-Computer Interface (BCI) systems. It combines classic signal processing with graph neural networks (EEG-ARNN) to keep accuracy high while using far fewer electrodes.

The project is organised around a set of reproducible Jupyter notebooks and helper modules. If you are joining the project for the first time, follow the steps below to set up your environment, download the data, and run the baseline and gated models.

---

## Highlights

- Works with the BCI Competition IV 2a dataset (PhysioNet mirror).
- Produces channel-reduced classifiers that maintain accuracy within ~2-3 percent of the full 64-electrode setup.
- Two training variants: baseline attention-based model and a gated attention model for ablation studies.
- Rich documentation under `docs/` covering theory, preprocessing mathematics, and experimental notes.

---

## 1. Prerequisites

- Operating system: Windows, macOS, or Linux.
- Python 3.10+ and `pip`.
- (Recommended) `conda` or `venv` for isolated environments.
- JupyterLab or Jupyter Notebook.
- At least 30 GB of free disk space for raw + processed data.

Optional but helpful:
- CUDA-capable GPU if you want to speed up training.
- `mne-bids` installation if you plan to extend the preprocessing scripts.

---

## 2. Environment Setup

```bash
# Clone or download the repository
git clone https://github.com/<your-user>/eeg-channel-selection.git
cd eeg-channel-selection

# Create a virtual environment (example with venv)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

If you are using Conda, create a new environment with `conda create -n eeg-channel-selection python=3.10` and activate it before installing the requirements.

---

## 3. Download the Dataset

All notebooks expect the BCI Competition IV 2a dataset to live under `data/physionet`. Start with:

1. Open `physionet_data_cleaning.ipynb`.
2. Run the download cells; they can fetch the data directly from PhysioNet (you need a PhysioNet account).
3. The notebook will generate `data/physionet/raw/` with `.edf` files and metadata CSVs in `data/physionet/derived/`.

You can also place existing downloads inside `data/physionet/raw/` and rerun the indexing cells to rebuild the metadata.

---

## 4. Run the Pipeline

Work through the notebooks in this order:

1. **Exploration** - `physionet-eda.ipynb`
   Inspect signals, events, and subject variability.
2. **Cleaning** - `physionet_data_cleaning.ipynb`
   Download data, flag unusable runs, and write `physionet_good_runs.csv`.
3. **Preprocessing** - `physionet_data_preprocessing.ipynb`
   Apply filtering, notch removal, bad-channel repair, and export `.fif` files.
4. **Training (Baseline)** - `physionet_training_baseline.ipynb`
   Run the default EEG-ARNN training loop and channel selection using attention scores.
5. **Training (Gated Variant)** - `physionet_training_gated.ipynb`
   Repeat training with an additional gating mechanism to compare performance and robustness.
6. **Results & Reporting** - `physionet_results_analysis.ipynb`
   Aggregate metrics, plot accuracy vs. electrode count, and export final figures.

Additional analysis lives under `analysis/` for side-by-side comparisons (`gated-vs-baseline.ipynb`, `physionet_method_comparison.ipynb`).

---

## 5. Notebook Walkthrough (File-wise Guide)

| File | Purpose | Typical Outputs |
| --- | --- | --- |
| `physionet-eda.ipynb` | Exploratory data analysis, channel inspection, event timing checks. | Overview plots saved to `analysis/` (optional) |
| `physionet_data_cleaning.ipynb` | Downloads raw EDF files, filters out noisy runs, builds `physionet_good_runs.csv`. | `data/physionet/raw/`, `data/physionet/derived/physionet_good_runs.csv` |
| `physionet_data_preprocessing.ipynb` | Applies filtering, downsampling, bad-channel interpolation, saves `.fif`. | `data/physionet/derived/preprocessed/`, `physionet_preprocessed_index.csv` |
| `physionet_training_baseline.ipynb` | Baseline EEG-ARNN training with attention-based channel scores. | `results/subject_results.csv`, `saved_models/baseline_*` |
| `physionet_training_gated.ipynb` | Adds gating network on top of attention to compare robustness and sparsity. | `results/retrain_results.csv`, `saved_models/gated_*` |
| `physionet_training_trial.ipynb` | Lightweight sandbox to test new hyperparameters or debug data issues without touching main runs. | Experiment-specific artifacts in `failed_trials/` |
| `physionet_results_analysis.ipynb` | Collects baseline and retrain CSVs, builds summary plots and markdown reports. | `results/channel_selection_analysis/` |
| `analysis/gated-vs-baseline.ipynb` | Direct comparison between the two training variants. | Comparative figures under `analysis/` |
| `analysis/physionet_method_comparison.ipynb` | Benchmarks alternative channel selection strategies. | CSV/plots in `analysis/` |

---

## 6. Supporting Modules and Docs

- `models.py` - EEG-ARNN architectures, attention heads, and gating blocks.
- `train_utils.py` - Training loops, PyTorch data utilities, checkpoint helpers.
- `update_config.py` - Command line helper to keep notebook configs in sync.
- `docs/COMPLETE_PIPELINE.md` - Narrative walk-through of the full pipeline.
- `docs/glossary.md` - Terminology quick reference for EEG and graph learning.
- `docs/preprocessing_pipeline_and_tools.md` - Detailed math for filtering, CAR, and ICA steps.

---

## 7. Key Outputs

Running the full pipeline produces:

- `results/subject_results.csv` - baseline accuracy per subject with all channels.
- `results/retrain_results.csv` - accuracy per subject after channel selection for each `k`.
- `results/channel_selection_analysis/` - plots, summaries, and Markdown reports.
- `saved_models/` - serialized PyTorch checkpoints for baseline and gated runs.
- `data/physionet/derived/preprocessed/` - cleaned `.fif` files ready for downstream experiments.

Expect roughly 6-7 hours on CPU for the baseline fast configuration (5 subjects, 10 epochs, 2-fold cross-validation). The gated variant roughly doubles training time; use the notebooks' configuration cells to trade off runtime and accuracy.

---

## 8. Repository Map

```
eeg-channel-selection/
|-- README.md
|-- requirements.txt
|-- models.py                   # EEG-ARNN architecture definitions
|-- train_utils.py              # Training loops, data loaders, logging utilities
|-- update_config.py            # Helper for reproducible experiment configs
|-- physionet-eda.ipynb
|-- physionet_data_cleaning.ipynb
|-- physionet_data_preprocessing.ipynb
|-- physionet_training_baseline.ipynb
|-- physionet_training_gated.ipynb
|-- physionet_training_trial.ipynb
|-- physionet_results_analysis.ipynb
|-- analysis/                   # Comparative notebooks and deep dives
|-- docs/                       # Extended documentation (math, glossary, pipeline overview)
|-- data/                       # Raw and processed EEG data (generated)
|-- results/                    # Exported figures, CSVs, and reports (generated)
|-- saved_models/               # Model checkpoints (generated)
|-- failed_trials/              # Debugging artifacts when runs abort (generated)
\-- trash/                      # Temporary scratch space (clean up regularly)
```

If you are unsure where to start, skim `docs/COMPLETE_PIPELINE.md` and `docs/glossary.md` - they explain terminology and the high-level flow.

---

## 9. Tips for New Contributors

- Keep notebooks lean: duplicate heavy experiments into `analysis/` so the main pipeline remains reproducible.
- Use the `config` dictionaries in the training notebooks to manage folds, subjects, and GPU vs. CPU runs.
- When adding new preprocessing steps, document them in `docs/preprocessing_pipeline_and_tools.md` and regenerate the downstream CSVs.
- Before a big experiment, clear `failed_trials/` to make debugging output easier to interpret.

---

## 10. Citation

If you use the channel-selection approach in publications, please cite:

```bibtex
@article{sun2023graph,
  title={Graph Convolution Neural Network Based End-to-End Channel Selection},
  author={Sun, Banghua and Liu, Zhiyuan and Wu, Zongqing and Mu, Chaoxu and Li, Tiancheng},
  journal={IEEE Transactions on Industrial Informatics},
  volume={19},
  number={9},
  pages={9314--9324},
  year={2023}
}
```

You can also cite this repository directly (add DOI or commit hash if minted).

---

## 11. Need Help?

- Browse the FAQs and walkthroughs in `docs/`.
- Open an issue describing your environment and the exact cell that failed.
- Reach out with pull requests if you improve preprocessing, add new selection strategies, or discover better hyperparameters.

Happy experimenting!
