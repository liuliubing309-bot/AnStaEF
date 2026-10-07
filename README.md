# AnStaEF: Anchoring and Statistics-Guided Expert Fusion for Multivariate Time Series Forecasting

This repository provides the PyTorch implementation of AnStaEF.

## Overview

AnStaEF combines an Anchor Stream for common temporal patterns with an Expert Stream for patch-specific local variations. The Expert Stream performs patch-level Top-k fusion over heterogeneous Fourier filtering, dilated TCN, and feature MLP candidate outputs, with base transformation parameters shared among candidates of the same type. Learned patch representations and local statistics (trend slope, volatility, and lag-1 autocorrelation) jointly guide expert selection and weighting.

Cross-Variable Context Exchange (CVCE) enriches patch representations before dual-stream processing, while Segment-Level Temporal Interaction (SLTI) models dependencies among segment representations after stream fusion.

## Requirements

Install [PyTorch](https://pytorch.org/get-started/locally/) for your Python version and CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is not included in `requirements.txt` and must be installed separately. The experiments in the paper use a single NVIDIA GeForce RTX 4090 GPU.

## Datasets

We evaluate AnStaEF on ETTh1, ETTh2, ETTm1, ETTm2, Exchange-Rate, Weather, Solar-Energy, and Electricity. Benchmark data sources include [Autoformer](https://github.com/thuml/Autoformer#get-started) and the [LSTNet dataset repository](https://github.com/laiguokun/multivariate-time-series-data).

Place the CSV files in `./data/`:

```text
data/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
    exchange_rate.csv
    weather.csv
    solar.csv
    electricity.csv
```

Each CSV must contain a `date` column followed by numeric variable columns. The supplied scripts expect a column named `OT` for all datasets except Solar-Energy, whose 137 variable columns are named `0` through `136`. Raw headerless text files must be converted to this CSV format before use; changing the filename extension alone is insufficient.

## Training and Evaluation

Run commands from the repository root using Bash (for example, Git Bash or WSL on Windows).

For ETTm1 with an input length of 96 and a prediction horizon of 96:

```bash
bash scripts/seq_len96_pred_len96/run_ettm1_96_96.sh
```

For Weather with the same input length and prediction horizon:

```bash
bash scripts/seq_len96_pred_len96/run_weather_96_96.sh
```

The 32 main experiment scripts cover eight datasets and four prediction horizons, with input length fixed to 96:

```text
scripts/
    seq_len96_pred_len96/
    seq_len96_pred_len192/
    seq_len96_pred_len336/
    seq_len96_pred_len720/
```

Dataset-specific settings are provided in the scripts. Shared defaults in `run.py` include a feature dimension of 256, eight candidate experts, Top-k = 2, and random seed 2024.

By default, `RUN_MODE = "train_test"` in `run.py` trains the model, restores the best validation checkpoint, and evaluates it on the test set. Checkpoints are saved to `./checkpoints/<setting>/checkpoint.pth`; predictions, targets, and metrics are saved to `./results/<setting>/`. The entries in `metrics.npy` are MAE, MSE, RMSE, MAPE, and MSPE, in that order.

To evaluate an existing checkpoint without training, set `RUN_MODE = "test_only"` in `run.py` and rerun the corresponding script with the same model and experiment settings.

## Code Structure

- `models/AnStaEF.py`: dual-stream framework, SLTI, and forecasting head.
- `layers/`: patch embedding, CVCE, Anchor Stream, base experts, statistics extraction, routing, and RevIN.
- `data_provider/`: dataset loading and preprocessing.
- `exp/exp_main.py`: training, validation, and testing.
- `utils/`: loss functions, metrics, time features, and training utilities.
- `scripts/`: experiment configurations and launch scripts.
