# Code, data and reproduction

Everything behind the numbers in the [summary](../README.md) and the [full report](../docs/REPORT_full.md).
Section numbers (§) refer to the full report. All commands run from this folder.

## Layout

```
constraint-study/
├── configs/          baseline.yaml (MLPerf Tiny reference), fast.yaml (validated sweep config)
├── src/
│   ├── data/         DCASE file discovery and labels
│   ├── features/     log-mel extraction, windowing, caching
│   ├── models/       autoencoder, quantisation (STE / LSQ), analog layer (analog.py)
│   ├── noise.py      device parameters; chip sampling isolated from training randomness
│   ├── tiling.py     30 × 30 tile budget
│   ├── train.py      fp32 and hardware-aware training
│   ├── evaluate.py   AUC / pAUC per machine ID
│   └── evaluate_analog.py, experiment.py
├── scripts/          data preparation, sweeps, diagnostics, figures
├── tests/            101 unit tests
└── results/          figures and aggregated CSVs (model checkpoints are not versioned)
```

## Setup

Developed on Windows 11 with an RTX 4060 Laptop GPU (8 GB) and Python 3.11.

```bash
conda create -n ailinear python=3.11 -y
conda activate ailinear
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/check_env.py
```

Install `torch` from the PyTorch index first: on Windows a plain `pip install torch` pulls the CPU-only
build. Pick the current CUDA tag at <https://pytorch.org/get-started/locally/>. `requirements-analog.txt`
(IBM aihwkit) is **not** needed: the study uses its own behavioural model, `src/models/analog.py`.

## Data

The ToyCar machine type of DCASE 2020 Task 2 (from ToyADMOS), exactly as the MLPerf Tiny benchmark uses it:

| Split | Source | Machine IDs | Clips |
|---|---|---|---|
| Training | development set + additional training set | 01–07 | 7,000, all normal |
| Test | development test set | 01–04 | 2,459: 1,400 normal, 1,059 anomalous |

Clips are 16 kHz mono, about 10 s. The official source is Zenodo
([development](https://zenodo.org/records/3678171), [additional training](https://zenodo.org/records/3727685)).
The same files are mirrored on Kaggle (ToyCar only, about 3.2 GB; needs a Kaggle API token):

```bash
kaggle datasets download -d daisukelab/dc2020task2 -p data/dc2020task2 --unzip
kaggle datasets download -d daisukelab/dc2020task2added -p data/dc2020task2added --unzip
```

The second dataset's title is misspelled ("Additinal"); its slug is `dc2020task2added`. If an archive
unpacks into an extra sub-folder, move `train/` and `test/` up one level:

```
data/
├── dc2020task2/
│   ├── train/    4000 wav   IDs 01–04, normal
│   └── test/     2459 wav   IDs 01–04, normal + anomalous
└── dc2020task2added/
    └── train/    3000 wav   IDs 05–07, normal
```

MLPerf Tiny trains **one model on all seven machine IDs**; without the additional training set only
4,000 clips remain and the reference number is not reproduced. Build the log-mel caches, one per number of
mel bands used in the sweeps (the script checks the layout and clip counts first):

```bash
python scripts/prepare_data.py --n-mels 6 10 16 30 32 64 128
```

## Reproducing the results

Scripts skip runs whose checkpoints already exist, so any command can be interrupted and restarted.
Times are approximate, on the GPU above.

| Result | Command | Time |
|---|---|---|
| Reference reproduction (§2.1) | `python -m src.train --config configs/baseline.yaml` then `python -m src.evaluate --config configs/baseline.yaml` | 35 min |
| Fast-configuration validation (§2.1) | `python -m src.train --config configs/fast.yaml` then `python -m src.evaluate --config configs/fast.yaml --compare c0_baseline` | 6 min |
| Figure 1, input dimensionality (§3.1) | `python scripts/sweep.py --track dim alloc chip --seeds 0 1 2` | 2.5 h |
| Leaky-integrator front end (§3.1) | `python scripts/probe_leaky.py --lr 0.0005 --tag lr5e-4 --arms fp32:frames fp32:tau512 fp32:tau128`, then `--tag lr5e-4_hwa --arms hwa6:frames hwa6:tau128` | 1 h |
| Figures 2–3, calibration, cross-evaluation (§3.2, §3.3, §4.2) | `N_SHARDS=3 bash scripts/run_v11.sh` | 5 h |
| fp32 noise-injection control (§3.3) | `bash scripts/run_noise_reg.sh` | 1 h |
| Figure 4, bit width (§3.4) | `bash scripts/run_ste_check.sh`, `run_fig4_stefix.sh`, `run_center_check.sh` (`N_SHARDS=3`) | 2 h |
| Revised device model (§3.5) | `N_SHARDS=3 bash scripts/run_v2.sh` | 5.5 h |
| Decision threshold (§4.3) | `python scripts/probe_operating_point.py` | 2 min |
| Figures from the CSVs | `python scripts/make_figures.py --fig 1 4`, then `python scripts/make_figures.py --tag stefix` (Figures 2–3) | seconds |
| Data for the one-page summary (`../site/data.js`) | `python scripts/export_site_data.py` | seconds |

Parallel scripts need about 1.4 GB of host memory per process; choose `N_SHARDS` from the free physical
memory, and stop the sub-shells as well as the Python processes when interrupting one. On an 8 GB laptop GPU
under Windows a single large allocation can fail even when memory is free, so feature caches larger than
0.5 GB stay in host memory and batches are streamed (`train.store_device: auto`).

## Tests

```bash
pytest -q
```

101 tests, including: the analog layer reduces exactly to `nn.Linear` when noise and quantisation are off;
device mismatch is frozen per chip while cycle-to-cycle noise is redrawn; chip sampling does not touch the
training random stream; tiled and dense matrix products agree; in the no-inter-layer-A/D mode hidden-layer
inputs stay continuous; the straight-through estimator passes the gradient on the top quantization level and a
quantized bias can learn a DC offset; fixed converter ranges calibrate in training, freeze in evaluation and
do not depend on batch composition; and the absolute, shot and proportional noise models scale as defined.
