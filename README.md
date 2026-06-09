# SEM-FAS Paper 2026

This repository contains the implementation and experiment pipeline for a
face anti-spoofing study built around a baseline MobileNetV3 model and an
Adaptive Quality-Bitrate Face Anti-Spoofing model, abbreviated as AQB-FAS.

The code was prepared for experiments on CelebA-Spoof and includes training,
evaluation, result aggregation, paper tables, figures, and a notebook dashboard.

## Main Features

- Baseline PAD model based on MobileNetV3.
- AQB-FAS model with latent dimensionality and bitrate controls.
- CelebA-Spoof CSV index builder with identity-disjoint train/validation split.
- Intel GPU support through PyTorch XPU, with CPU fallback through config edits.
- Resumable training using `last.pt` or `best.pt`.
- Test evaluation with validation-selected threshold.
- Paper artifacts: summary CSV, LaTeX table, training curves, confusion matrices,
  utility-bitrate figure, and ablation figure.
- Jupyter notebook for visual result inspection.

## Repository Layout

```text
configs/                 Experiment YAML configs
data/raw/                Raw CelebA-Spoof dataset, not committed
data/processed/          Generated train/val/test CSVs, not committed
notebooks/               Result dashboard notebook
outputs/figures/         Paper figures, small generated PNGs
outputs/results/         Aggregated result CSV
outputs/runs/            Checkpoints and per-run logs, not committed
outputs/tables/          Paper-ready CSV and LaTeX tables
paper/                   Paper-related files
scripts/                 Data, pipeline, plotting, and artifact scripts
src/                     Dataset, model, training, testing, metrics code
```

Large files are intentionally excluded from Git:

- CelebA-Spoof raw images and metadata under `data/raw/`
- Processed CSV indexes under `data/processed/`
- Training checkpoints under `outputs/runs/`
- Zip archives and local logs

## Environment

The experiments were run with Miniconda on Windows using an Intel Arc GPU
through PyTorch XPU.

Create and activate an environment:

```powershell
conda create -n semcomfas python=3.11 -y
conda activate semcomfas
pip install -r requirements.txt
```

For Intel GPU/XPU, install a PyTorch build that supports XPU for your system.
After installation, verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available() if hasattr(torch, 'xpu') else False)"
```

Expected for Intel GPU usage:

```text
True
```

The default configs use:

```yaml
device: xpu
```

If your machine does not have Intel XPU support, edit the config files and use:

```yaml
device: cpu
```

For NVIDIA CUDA users, install the proper CUDA PyTorch build and set:

```yaml
device: cuda
```

## Dataset Setup

This project expects CelebA-Spoof in:

```text
data/raw/CelebA-Spoof
```

Expected structure:

```text
data/raw/CelebA-Spoof/
  Data/
  metas/
  README
```

The dataset is not included in this repository. Place or extract CelebA-Spoof
manually before building the indexes.

## Build Data Indexes

Generate the debug and full train/validation/test CSV files:

```powershell
python scripts\build_index.py --full
```

This writes:

```text
data/processed/debug_2k.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

The train/validation split is identity-disjoint to reduce leakage between
training and validation subjects.

## Smoke Tests

Run these before long training:

```powershell
python scripts\check_metrics.py
python scripts\check_models.py
python scripts\check_train_loop.py --model aqb_fas --device cpu --epochs 1
```

For a quick XPU debug run:

```powershell
python src\train.py --config configs\debug_aqb_z64_b8.yaml
```

## Training

Train the baseline:

```powershell
python src\train.py --config configs\baseline_mbv3.yaml
```

Train the main AQB-FAS model:

```powershell
python src\train.py --config configs\aqb_z64_b8.yaml
```

Resume an interrupted run:

```powershell
python src\train.py --config configs\aqb_z64_b8.yaml --resume
```

Each run writes to its configured run directory:

```text
outputs/runs/<run_name>/
  best.pt
  last.pt
  history.csv
  best_metrics.json
  last_metrics.json
  config.json
```

Checkpoints are not meant to be committed to Git.

## Testing

Evaluate a trained model on the test split:

```powershell
python src\test.py --config configs\baseline_mbv3.yaml
python src\test.py --config configs\aqb_z64_b8.yaml
```

Testing loads `best.pt` and uses the validation-selected threshold from
`best_metrics.json`. Test metrics are written to:

```text
outputs/runs/<run_name>/test_metrics.json
```

## Full Pipeline

Run the default paper pipeline, baseline first and then AQB-FAS:

```powershell
python scripts\run_pipeline.py
```

Run only the main AQB-FAS config and resume if interrupted:

```powershell
python scripts\run_pipeline.py --configs configs\aqb_z64_b8.yaml --resume
```

Useful flags:

```text
--rebuild-index      Rebuild data/processed CSV files before training
--resume             Resume training from last.pt, or best.pt if last.pt is missing
--skip-test          Skip test evaluation
--skip-artifacts     Skip table and figure generation
```

After training and testing, the pipeline generates paper artifacts.

## Paper Artifacts

Regenerate all tables and figures from existing run outputs:

```powershell
python scripts\make_artifacts.py
```

Generated files:

```text
outputs/results/summary.csv
outputs/tables/main_results.csv
outputs/tables/main_results.tex
outputs/figures/training_curves_baseline_mbv3.png
outputs/figures/training_curves_aqb_z64_b8.png
outputs/figures/confusion_matrices.png
outputs/figures/utility_bitrate.png
outputs/figures/ablation_acer.png
```

## Result Dashboard

Open the notebook:

```text
notebooks/results_dashboard.ipynb
```

The notebook loads the generated CSV, JSON, and PNG artifacts and shows:

- Main result table
- Baseline vs AQB-FAS comparison
- Validation/test gap
- Test confusion matrices
- Precision, recall, and F1 score
- Training curves
- Utility-bitrate plot
- Ablation plot

## Current Main Test Results

The current generated table reports:

| Method | Latent bits | Test ACC (%) | Test Precision (%) | Test Recall (%) | Test F1 (%) | Test AUC (%) | Test ACER (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline MobileNetV3 | Full image | 84.349 | 99.889 | 77.836 | 87.494 | 99.447 | 11.185 |
| AQB-FAS z=64, b=8 | 512 | 80.201 | 99.985 | 71.863 | 83.623 | 99.519 | 14.081 |

Validation metrics are more optimistic than test metrics, so paper discussion
should focus on the test set for the main claim.

## Reproducibility Notes

- Positive class is `spoof` with label `1`; live is label `0`.
- APCER is spoof predicted as live divided by total spoof samples.
- BPCER is live predicted as spoof divided by total live samples.
- ACER is `(APCER + BPCER) / 2`.
- Precision, recall, and F1 are computed for the spoof class.
- Threshold selection is done on validation data, not on the test set.
- Default random seed is `42`.

## Git Notes

Do not commit the raw dataset or checkpoints. Recommended files to commit are:

- `src/`
- `scripts/`
- `configs/`
- `notebooks/results_dashboard.ipynb`
- Small paper artifacts in `outputs/results/`, `outputs/tables/`, and
  `outputs/figures/`

Avoid committing:

- `data/`
- `outputs/runs/`
- `*.pt`
- `*.zip`
- local IDE settings and terminal logs
