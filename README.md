# Machine Acoustic Anomaly Detection on a 30 × 30, 6-bit Analog MAC Array

**A simulation study of what it takes to run an anomaly-detection model on an analog in-memory-compute
chip with no A/D conversion between hidden layers.**

[![The study on one page: 380 tiles to 6, the signal chain on the chip, and what each block must guarantee](site/poster.png)](site/index.html)

*The same page with interactive plots of every simulated chip: open [`site/index.html`](site/index.html) in a
browser. It runs locally, with no server or install.*

The MLPerf Tiny anomaly-detection reference model — ten fully connected layers,
`640 → [128]×4 → 8 → [128]×4 → 640` — needs 380 crossbar tiles of 30 × 30. This study compresses it to
a six-layer network, `30 → [30]×2 → 4 → [30]×2 → 30`, that fits in **6 tiles** (one per layer), runs it
on a behavioural model of a 6-bit analog MAC array — per-chip
device mismatch, weight-programming error, signal-path noise, and different placements of A/D and D/A
converters — and measures what each hardware constraint costs.

> **Simulation only — nothing here has been verified on silicon.** The hardware constraints are taken
> from published test-chip specifications for analog in-memory inference. Every modelling assumption is
> listed in [assumptions.md](assumptions.md).

## Results at a glance

Task: ToyCar, DCASE 2020 Task 2 (unsupervised machine-sound anomaly detection).
Metric: AUC, mean ± std over training seeds × 10 simulated chips.

> **Version 2 (23 September 2026).** Finding 6 of version 1 was wrong — a bug in the quantization-aware
> training, not a hardware limit. Every quantized result has been retrained on the corrected code and
> re-tested under a revised device model; see *Changes since version 1* in the
> [full report](docs/REPORT_full.md).

| # | Finding | Key numbers |
|---|---|---|
| 1 | A 6-tile model is viable on the simulated analog array | 0.722 ± 0.012 on analog hardware vs 0.720–0.727 in fp32; ≈ 78 % of the above-chance discrimination of the 380-tile reference (0.785); 0.70–0.73 under every revised device model |
| 2 | Spend the 30-dimensional input budget on time, not frequency | 6 bands × 5 frames beats 30 bands × 1 frame by +0.068 (t = 4.2); leaky integrators with τ ≤ 128 ms match stored frames |
| 3 | Programming error up to 3 % is nearly free — with hardware-aware training | 0.714–0.735 at σ_prog ≤ 3 %; without it, chips worse than random appear from 1 % |
| 4 | Where the converters sit is not the constraint; the noise of the signal path is | Removing inter-layer A/D: −0.010 / −0.012 at 0.24 / 0.47 LSB, +0.007 with fixed converter ranges; both placements fail at ≈ 1 LSB |
| 5 | Activation noise during training decides robustness — it is a regulariser | 0.646 → 0.733 on a quiet chip; +0.014 / +0.027 in plain fp32 at 4 / 8 %; calibration recovers 0.018 of a 0.089 gap |
| 6 | The converters, not the weights, set the bit-width floor *(corrected in version 2)* | Weights and biases at 4 bits: −0.010 / +0.005; converters at 4 bits: 0.570, unless a per-channel offset precedes the D/A — then 4-bit W/S/B gives 0.723 (6 bits: 0.722) |

![Figure 2](constraint-study/results/figures/fig2_noise_stefix.png)

*Hardware-aware training (blue) keeps AUC close to fp32 as weight-programming error grows; without it
(red), the worst simulated chip falls below random guessing.*

The method, all four figures, per-condition tables, the chip-design implications, the draft handoff
interface between the ML and hardware sides, and the limitations are in the
**[full report](docs/REPORT_full.md)** (Chinese: [REPORT_full-zh.md](docs/REPORT_full-zh.md)). A five-minute
version — the question, the trade-offs and the conclusions in plain language — is **[REPORT.md](REPORT.md)**.
Section numbers (§) in this README and in `assumptions.md` refer to the full report.

## Repository layout

```
.
├── README.md             this file
├── REPORT.md             short report, version 2 (five minutes, plain language)
├── docs/REPORT_full.md   full report, version 2; REPORT_full-zh.md is its Chinese translation;
│                         version 1 as sent: docs/REPORT_sent.md
├── assumptions.md        every modelling assumption, and open questions for the hardware team
├── site/                 one-page summary: index.html (interactive), poster.png, data.js (generated)
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
    ├── tests/            101 unit tests
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
| Reference reproduction (report §2.1) | `python -m src.train --config configs/baseline.yaml` then `python -m src.evaluate --config configs/baseline.yaml` | 35 min |
| Fast-configuration validation (§2.1) | `python -m src.train --config configs/fast.yaml` then `python -m src.evaluate --config configs/fast.yaml --compare c0_baseline` | 6 min |
| Figure 1 — input dimensionality (§3.1) | `python scripts/sweep.py --track dim alloc chip --seeds 0 1 2` | 2.5 h |
| Leaky-integrator front end (§3.1) | `python scripts/probe_leaky.py --lr 0.0005 --tag lr5e-4 --arms fp32:frames fp32:tau512 fp32:tau128`, then `--tag lr5e-4_hwa --arms hwa6:frames hwa6:tau128` | 1 h |
| Figures 2–3, BN calibration, cross-evaluation (§3.2, §3.3, §4.2) | `N_SHARDS=3 bash scripts/run_v11.sh` | 5 h |
| fp32 noise-injection control (§3.3) | `bash scripts/run_noise_reg.sh` | 1 h |
| Figure 4 — bit width (§3.4) | `bash scripts/run_ste_check.sh`, `run_fig4_stefix.sh`, `run_center_check.sh` (`N_SHARDS=3`) | 2 h |
| Revised device model (§3.5) | `N_SHARDS=3 bash scripts/run_v2.sh` | 5.5 h |
| Decision threshold (§4.3) | `python scripts/probe_operating_point.py` | 2 min |
| Figures from the CSVs | `python scripts/make_figures.py` (version 1) and `python scripts/make_figures.py --tag stefix` (Figures 2–3, version 2; Figure 4 v2 is written with Figure 4) | seconds |
| Robustness table, version 1 → 2 | `python scripts/robustness_table.py` | seconds |
| Data for the one-page summary (`site/data.js`) | `python scripts/export_site_data.py` | seconds |

Version-1 results are reproduced with the scripts' defaults (no `--tag`): `sweep_noise.py --seeds 0 1 2 3`,
`bash scripts/run_day4.sh`, and `sweep_bits.py --train-sigma-read 0` for the bit-width run without
activation noise. They train with the corrected straight-through estimator, so the quantized results will
differ from those in version 1.

Several parallel processes need about 1.4 GB of host memory each; choose `N_SHARDS` from the free physical
memory. Stopping a parallel script requires stopping its sub-shells as well as the Python processes.

On an 8 GB laptop GPU under Windows, a single large GPU allocation can fail even when memory is free.
The training script therefore keeps feature caches larger than 0.5 GB in host memory and streams
batches (`train.store_device: auto` in the configs).

## Tests

```bash
pytest -q
```

101 tests, including: the analog layer reduces exactly to `nn.Linear` when noise and quantisation are
off; device mismatch is frozen per chip while cycle-to-cycle noise is redrawn; chip sampling does not
touch the training random stream; tiled and dense matrix products agree; in the no-inter-layer-A/D
mode hidden-layer inputs stay continuous; the straight-through estimator passes the gradient on the top
quantization level, and a quantized bias can learn a DC offset (the version-1 bug); fixed converter ranges
calibrate in training, freeze in evaluation and do not depend on batch composition; and the absolute, shot
and proportional noise models scale as defined, with absolute noise adding per input tile.
