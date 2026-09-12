# Machine Acoustic Anomaly Detection on a 30 × 30, 6-bit Analog MAC Array

**A simulation study of what it takes to run an anomaly-detection model on an analog in-memory-compute
chip with no A/D conversion between hidden layers.**

The MLPerf Tiny anomaly-detection reference model needs 380 crossbar tiles of 30 × 30. This study
compresses it to **6 tiles**, runs it on a behavioural model of a 6-bit analog MAC array — per-chip
device mismatch, weight-programming error, signal-path noise, and different placements of A/D and D/A
converters — and measures what each hardware constraint costs.

> **Simulation only — nothing here has been verified on silicon.** The hardware constraints are taken
> from published test-chip specifications for analog in-memory inference. Every modelling assumption is
> listed in [assumptions.md](assumptions.md).

## Results at a glance

Task: ToyCar, DCASE 2020 Task 2 (unsupervised machine-sound anomaly detection).
Metric: AUC, mean ± std over training seeds × 10 simulated chips.

| # | Finding | Key numbers |
|---|---|---|
| 1 | A 6-tile model is viable on the simulated analog array | 0.729 ± 0.012 on analog hardware vs 0.720–0.727 in fp32; ≈ 80 % of the above-chance discrimination of the 380-tile reference (0.785) |
| 2 | Spend the 30-dimensional input budget on time, not frequency | 6 bands × 5 frames beats 30 bands × 1 frame by +0.068 (t = 4.2) |
| 3 | Programming error up to 3 % is nearly free — with hardware-aware training | 0.718–0.728 at σ_prog ≤ 3 %; without it, chips worse than random appear from 1 % |
| 4 | Removing inter-layer A/D costs ≤ 0.01 while signal-path noise stays below ≈ 0.5 LSB — an accuracy budget for the signal path | Paired difference +0.003 / −0.010 at 0.24 / 0.47 LSB; both architectures fail at ≈ 1 LSB |
| 5 | The no-inter-layer-A/D architecture needs activation noise during training, and per-chip calibration is no substitute | 0.665 → 0.738 on a quiet chip; 11 min of per-chip calibration recovers 0.022 of a 0.083 gap |
| 6 | 6-bit W/S/B is sufficient; 4-bit collapses | 6 / 8 / 10 bits: 0.729 / 0.727 / 0.727; 4 bits: 0.449 |

![Figure 2](constraint-study/results/figures/fig2_noise.png)

*Hardware-aware training (blue) keeps AUC close to fp32 as weight-programming error grows; without it
(red), the worst simulated chip falls below random guessing.*

The method, all four figures, per-condition tables, the chip-design implications, the draft handoff
interface between the ML and hardware sides, and the limitations are in **[REPORT.md](REPORT.md)**.

## Repository layout

```
.
├── README.md             this file
├── REPORT.md             final report
├── assumptions.md        every modelling assumption, and open questions for the hardware team
└── constraint-study/
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
    ├── tests/            76 unit tests
    └── results/          figures and aggregated CSVs (model checkpoints are not versioned)
```

## Setup

Developed on Windows 11 with an RTX 4060 Laptop GPU (8 GB) and Python 3.11. All commands below run from
`constraint-study/`.

```bash
conda create -n ailinear python=3.11 -y
conda activate ailinear
cd constraint-study
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/check_env.py
```

Install `torch` from the PyTorch index first: on Windows a plain `pip install torch` pulls the CPU-only
build. Pick the current CUDA tag at <https://pytorch.org/get-started/locally/>.

`requirements-analog.txt` (IBM aihwkit) is **not** needed: the study uses its own behavioural model,
`src/models/analog.py`.

## Data

The study uses the ToyCar machine type of DCASE 2020 Task 2 (from ToyADMOS), exactly as the MLPerf Tiny
benchmark does:

| Split | Source | Machine IDs | Clips |
|---|---|---|---|
| Training | development set + additional training set | 01–07 | 7,000, all normal |
| Test | development test set | 01–04 | 2,459: 1,400 normal, 1,059 anomalous |

Clips are 16 kHz mono, about 10 s. The official source is Zenodo
([development](https://zenodo.org/records/3678171), [additional training](https://zenodo.org/records/3727685)).
If Zenodo is unreachable, the same files are mirrored on Kaggle (ToyCar only, about 3.2 GB; needs a
Kaggle API token):

```bash
kaggle datasets download -d daisukelab/dc2020task2 -p data/dc2020task2 --unzip
kaggle datasets download -d daisukelab/dc2020task2added -p data/dc2020task2added --unzip
```

The second dataset's title is misspelled ("Additinal"); its slug is `dc2020task2added`. The expected
layout is below — if an archive unpacks into an extra sub-folder, move `train/` and `test/` up one level.
Paths are set in `configs/baseline.yaml` and `configs/fast.yaml`.

```
constraint-study/data/
├── dc2020task2/
│   ├── train/    4000 wav   IDs 01–04, normal
│   └── test/     2459 wav   IDs 01–04, normal + anomalous
└── dc2020task2added/
    └── train/    3000 wav   IDs 05–07, normal
```

MLPerf Tiny trains **one model on all seven machine IDs** (the DCASE 2020 baseline trains one model per
ID). Without the additional training set only 4,000 clips remain and the reference number is not
reproduced.

Build the log-mel caches — one per number of mel bands used in the sweeps. The script checks the layout
and the clip counts first:

```bash
python scripts/prepare_data.py --n-mels 6 10 16 30 32 64 128
```

## Reproducing the results

Scripts skip runs whose checkpoints already exist, so any command can be interrupted and restarted.
Times are approximate, on the GPU above.

| Result | Command | Time |
|---|---|---|
| Reference reproduction (report §3.1) | `python -m src.train --config configs/baseline.yaml` then `python -m src.evaluate --config configs/baseline.yaml` | 35 min |
| Fast-configuration validation (§2.3) | `python -m src.train --config configs/fast.yaml` then `python -m src.evaluate --config configs/fast.yaml --compare c0_baseline` | 6 min |
| Figure 1 — input dimensionality (§3.2) | `python scripts/sweep.py --track dim alloc chip --seeds 0 1 2` | 2.5 h |
| Figure 2 — programming error (§3.3) | `python scripts/sweep_noise.py --seeds 0 1 2 3` | 3 h |
| Figures 3–4 — A/D placement, bit width (§3.4–3.5) | `PY=python bash scripts/run_day4.sh` (4 parallel processes) | 3.5 h |
| Diagnostics (§3.4) | `python scripts/probe_train_noise.py` and `python scripts/probe_bn_calib.py` | 5 min |
| All figures from the CSVs | `python scripts/make_figures.py` | seconds |

The earlier bit-width run without activation noise (§3.5) is reproduced with
`python scripts/sweep_bits.py --train-sigma-read 0`.

On an 8 GB laptop GPU under Windows, a single large GPU allocation can fail even when memory is free.
The training script therefore keeps feature caches larger than 0.5 GB in host memory and streams
batches (`train.store_device: auto` in the configs).

## Tests

```bash
pytest -q
```

76 tests, including: the analog layer reduces exactly to `nn.Linear` when noise and quantisation are
off; device mismatch is frozen per chip while cycle-to-cycle noise is redrawn; chip sampling does not
touch the training random stream; tiled and dense matrix products agree; and in the no-inter-layer-A/D
mode hidden-layer inputs stay continuous.
