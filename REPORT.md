# Feasibility of a 30 × 30, 6-bit Analog MAC Array for Machine Acoustic Anomaly Detection

*A simulation study — September 2026*
*Author: KaiwenLin — KaiwenLin@utexas.edu*

> **Simulation only — nothing here has been verified on silicon.** Hardware parameters are inferred
> from published test-chip specifications for analog in-memory inference, or taken from the literature;
> each one is listed with its status in [assumptions.md](assumptions.md). The code, configurations and
> aggregated results behind every number are in [`constraint-study/`](constraint-study/); the
> [README](README.md) explains how to reproduce them.

---

## Summary

The MLPerf Tiny anomaly-detection reference model occupies **380 tiles** of a 30 × 30 crossbar, and one of its layers alone needs 110. This study compresses it to **6 tiles**, one per layer, and evaluates it on a behavioural model of a 6-bit analog MAC array with no A/D or D/A conversion between hidden layers:
10 simulated chips, at least three training seeds per condition. Task: ToyCar machine-sound anomaly detection (DCASE 2020 Task 2).

1. **Six tiles are enough.** On the simulated array (6-bit W/S/B, 3 % programming error, 2 % device
   mismatch, signal-path noise, no inter-layer A/D) the compressed model reaches **AUC 0.729 ± 0.012** —
   level with the same network in fp32 (0.720–0.727) and about 80 % of the above-chance discrimination
   of the 380-tile reference (0.785 ± 0.008). Every layer fits one tile, so multi-tile cascading is not
   on its critical path. (§2.2, §3.1)
2. **Spend the input budget on time, not frequency.** At equal dimension, 6 mel bands × 5 frames beats
   30 bands × 1 frame by **+0.068 AUC** (t = 4.2). The 30 input lines are the scarce resource, and an
   analog front end with configurable time constants can supply temporal context before the D/A without
   consuming them. (§3.1)
3. **Hardware-aware training is mandatory, and its payoff is yield.** With it, AUC holds at 0.718–0.728
   up to 3 % programming error — so a program-verify tolerance of ≈ 3 % is enough, and tighter
   programming only costs test time. Without it, the worst of ten simulated chips is below random from
   1 %, and the chip-to-chip spread is up to 6.6× wider. (§3.2)
4. **Removing inter-layer A/D costs ≤ 0.01 AUC while signal-path noise stays below ≈ 0.5 LSB**
   (σ_read ≲ 8 % of the mean signal, SNR ≳ 22 dB against it). At ≈ 1 LSB both placements fail together:
   the binding constraint is the noise of the signal path, not where the converters sit. (§3.3)
5. **Activation noise during training decides robustness — not the noise of the deployed chip.**
   Without it the no-inter-layer-A/D model loses 0.073 AUC even on a perfectly quiet chip, and per-chip
   calibration does not recover it. The training recipe belongs in the specification alongside the
   weights. (§3.3, §4.2)
6. **6-bit W/S/B is sufficient; 4-bit collapses.** 6, 8 and 10 bits are indistinguishable
   (0.729 / 0.727 / 0.727); 4 bits falls below chance (0.449). 5 bits was not tested, so the margin of
   the 6-bit specification is unknown. (§3.4)

---

## 1. Target and task

| Constraint | Value |
|---|---|
| Weight array | 30 × 30 |
| W / S / B precision | 6 bit each |
| A/D, D/A in hidden layers | none |
| Weight storage | in memory (EEPROM or ReRAM); weights fixed at inference |
| Operators | fully connected, batch normalisation, scalar multiply — no convolution |
| Operation | asynchronous, clockless, current mode |

The absence of convolution decides the model class; the 30 × 30 array decides everything else. Both are
parameters throughout the code, so every result can be recomputed for a different array (A1).

**Task:** unsupervised machine-condition monitoring from sound — training on normal clips only, scoring by reconstruction error. Benchmark: the ToyCar machine type of DCASE 2020 Task 2, as used by MLPerf Tiny.

---

## 2. Method

### 2.1 Reference model, data and protocol

The starting point is the MLPerf Tiny anomaly-detection reference: a fully connected autoencoder
`640 → [128]×4 → 8 → [128]×4 → 640` over log-mel features (128 bands, 5-frame sliding window,
n_fft 1024, hop 512), with a clip scored by the mean reconstruction MSE over its windows. Being
convolution-free, it maps directly onto a MAC array. Data follow the MLPerf Tiny protocol: one model
trained on machine IDs 01–07 (7,000 normal clips), evaluated on IDs 01–04 (1,400 normal, 1,059
anomalous). Reproduced here at AUC 0.7804 / pAUC 0.6737 against the published 0.8009 / 0.6722 — pAUC
matches to +0.0015, and the AUC gap is attributed to Keras ↔ PyTorch differences and was not chased
(A9).

**Protocol.** AUC is the primary metric; pAUC (FPR ≤ 0.1) is recorded for every run and follows the same trends. Every reported number is a mean ± std over (training seeds × 10 simulated chips), with at least 3 seeds per condition — single-seed results proved unreliable, with seed-to-seed std up to 0.065.
A run whose final *validation* loss exceeds twice the median of comparable runs is excluded; test AUC never enters that filter, which removed 1 of 24, 1 of 54 and 0 of 15 runs. Sweeps use a faster training configuration, accepted only after it matched the reference on both metrics at the reference operating point (ΔAUC −0.003, ΔpAUC −0.008, 5.6× faster; A8). All values are in `constraint-study/results/*.csv`.

### 2.2 Mapping the network onto tiles

A weight matrix `[out, in]` occupies `ceil(out / 30) × ceil(in / 30)` tiles (`src/tiling.py`), but the
two tiling directions are not equally cheap:

- **Output-direction tiling** splits a layer across tiles computing disjoint outputs. They do not interact; the cost is area and programming time.
- **Input-direction tiling** splits a dot product into partial sums that must be added. In the analog domain that addition is a wire — Kirchhoff's current law — which is why crossbars are attractive. But
  every contributing tile adds its own mismatch and read noise to the same summing node.
  **Input-direction tiling is where accuracy is spent.**

| Layer | Reference | Tiles (out × in) | Target | Tiles |
|---|---|---|---|---|
| 1 | FC 640 → 128 | **110** (5 × 22) | FC 30 → 30 | 1 |
| 2–4 | FC 128 → 128 | 25 each (5 × 5) | FC 30 → 30 | 1 |
| bottleneck in | FC 128 → 8 | 5 | FC 30 → 4 | 1 |
| bottleneck out | FC 8 → 128 | 5 | FC 4 → 30 | 1 |
| 6–8 | FC 128 → 128 | 25 each | FC 30 → 30 | 1 |
| output | FC 128 → 640 | **110** (22 × 5) | FC 30 → 30 | 1 |
| **Total** | | **380** | | **6** |
| Largest layer | | 110 | | **1** |
| Parameters | | 267,928 | | 4,242 |

The reference model's first layer needs 22 tiles in the input direction alone: 22 partial-sum currents on every output node, each carrying its own mismatch and noise. The target model tiles in neither direction, so multi-tile cascading — the one hardware capability this study could not confirm (A7) — is not required to run it.

### 2.3 Behavioural model of the array

Each fully connected layer is replaced by a simulated analog layer (`src/models/analog.py`):

- **Quantisation** of weights, signals and biases at 6 bits, with weight scales **per 30 × 30 tile** (weights are mapped onto each tile's conductance range). Straight-through estimator during training.
- **Device-to-device mismatch** (σ_d2d = 2 %): a multiplicative factor per weight, drawn once per chip and frozen. Redrawn every step during training so the model cannot specialise to one chip; chips come
  from a random stream independent of the training seeds.
- **Programming / cycle-to-cycle error** (σ_prog): multiplicative per weight, redrawn every inference.
- **Signal-path read noise** (σ_read): additive Gaussian on each layer's output,
  std = σ_read · mean|W·x|. At 6 bits the measured noise-to-LSB ratio is ≈ 5.93 σ_read, so 4 / 8 / 17 / 34 % correspond to ≈ 0.24 / 0.47 / 1.0 / 2.0 LSB. This is a **stress range, not a device
  parameter** (A15).
- **Converter placement**, assigned per layer by its position in the network:

| Mode | Signals quantised at | Conversions on the signal path (6 layers) |
|---|---|---|
| `none` | first layer's input, last layer's output | 2 |
| `per_layer` | input and output of every layer | 12 |
| `every_input` | input of every layer | 6 |

- **Hardware-aware training (HWA):** the network is trained with the analog layers in place — quantisation and noise in every forward pass — then deployed to 10 simulated chips without retraining.

Batch normalisation is kept as a separate affine stage, assumed to be a programmable gain and offset (A5). The hidden non-linearity is ReLU (A4).

---

## 3. Results

### 3.1 What fits (Figure 1)

fp32 training, 3 seeds.

![Figure 1](constraint-study/results/figures/fig1_dimension.png)

| | Input | Tiles (largest layer) | Parameters | AUC |
|---|---|---|---|---|
| Reference | 128 bands × 5 frames = 640 | 380 (110) | 267,928 | 0.7847 ± 0.0084 |
| **Target** | 6 bands × 5 frames = 30 | **6 (1)** | **4,242** | 0.7268 ± 0.0117 |

The target keeps 79.7 % of the reference's above-chance discrimination with 63× fewer tiles and parameters, and every layer fits a single tile. Frequency resolution costs about 0.021 AUC per halving (640 / 320 / 160 / 80 dimensions give 0.785 / 0.762 / 0.742 / 0.721).

**How to spend 30 input lines** (hidden width 30):

| Bands × frames | 6 × 5 | 10 × 3 | 16 × 2 | 30 × 1 | 32 × 1 |
|---|---|---|---|---|---|
| AUC | **0.727 ± 0.012** | 0.717 ± 0.011 | 0.714 ± 0.015 | 0.659 ± 0.026 | 0.662 ± 0.022 |

Every configuration with at least two frames scores ≥ 0.713; both single-frame configurations score ≤ 0.662. 6 × 5 versus 30 × 1 is +0.068 (pooled SE 0.016, t = 4.2). This is a statement about the front end rather than the network: the 30 input lines are the scarce resource, and they are better spent on a few bands carrying several time samples than on a finely resolved instantaneous spectrum. Envelope or leaky-integrator outputs at a handful of configurable time constants would supply that context before the D/A, without consuming input lines.

### 3.2 Programming error and hardware-aware training (Figure 2)

Target network, σ_d2d = 2 %, 4 seeds × 10 chips. **C1** fp32 training and inference (upper bound); **C2** fp32 training, analog inference; **C3** HWA; **C4** HWA plus a 6-bit quantisation at every layer output.

![Figure 2](constraint-study/results/figures/fig2_noise.png)

| σ_prog | C2 (no HWA) | C3 (HWA) | Worst chip, C2 | Worst chip, C3 |
|---|---|---|---|---|
| 0 % | 0.689 ± 0.069 | **0.719 ± 0.010** | 0.497 | 0.700 |
| 1 % | 0.662 ± 0.081 | **0.728 ± 0.014** | 0.427 | 0.698 |
| 2 % | 0.649 ± 0.093 | **0.721 ± 0.019** | 0.423 | 0.686 |
| 3 % | 0.628 ± 0.076 | **0.718 ± 0.017** | 0.425 | 0.685 |
| 5 % | 0.587 ± 0.072 | **0.702 ± 0.033** | 0.404 | 0.596 |
| 10 % | 0.549 ± 0.082 | **0.644 ± 0.044** | 0.344 | 0.558 |

C1 = 0.720 ± 0.018; C4 − C3 stays within ±0.01 everywhere.

- **Up to 3 % programming error is nearly free with HWA** (0.718–0.728 against 0.720 for fp32; read as "within about 0.01", given A16). 5 % costs about 0.017, 10 % about 0.075. Tighter programming buys no accuracy but costs programming time, which scales with weight count and therefore with test cost.
- **The mean is the wrong number to read.** Without HWA it looks survivable at 0.63–0.69 while the worst simulated chip is below random from 1 % programming error — a part that ships at 0.43 is a returned
  part. HWA narrows the chip-to-chip spread by up to 6.6×.

*Placement note: these runs predate the `none` mode and quantise the input of every layer, so C3 sits closer to conventional in-memory compute than to the target architecture; C3 vs C4 shows only that an extra quantisation is free. Removing inter-layer conversion is tested in §3.3, and at the nominal operating point the two agree (0.718 ± 0.017 vs 0.729 ± 0.012).*

### 3.3 Inter-layer A/D and signal-path noise (Figure 3)

Both placements, each trained with HWA under its own conditions; σ_prog = 3 %, σ_d2d = 2 %, 3 seeds ×
10 chips. Differences are paired by seed.

![Figure 3](constraint-study/results/figures/fig3_adc.png)

| σ_read | ≈ noise / LSB | `none` | `per_layer` | Paired difference |
|---|---|---|---|---|
| 0 | 0 | 0.650 ± 0.058 | 0.732 ± 0.021 | −0.082 ± 0.057 (training recipe, below) |
| 4 % | 0.24 | 0.729 ± 0.013 | 0.726 ± 0.014 | **+0.003 ± 0.004** |
| 8 % | 0.47 | 0.720 ± 0.010 | 0.729 ± 0.009 | **−0.010 ± 0.002** |
| 17 % | 1.0 | 0.524 ± 0.070 | 0.574 ± 0.005 † | −0.002 ± 0.004 † |
| 34 % | 2.0 | 0.491 ± 0.008 | 0.472 ± 0.035 | +0.019 ± 0.039 |

† One `per_layer` run removed by the convergence filter; 2 seeds.

**Below ≈ 0.5 LSB, removing the inter-layer converters costs at most 0.01 AUC.** An inter-layer A/D does regenerate part of the noise — visible at 0.47 LSB — and the effect grows with depth (paired difference −0.002 / −0.010 / −0.016 at 1 / 2 / 4 blocks, with depth confounded by parameter count). At ≈ 1 LSB both placements fail together, and at 2 LSB both are at chance. **What sets the limit is the noise of the signal path, not the placement of the converters**, and the transition is not gentle: between 0.47 and 1.0 LSB, AUC falls from 0.72 to 0.52.

That gives the trade a price on both sides. Published component-level figures for ultra-low-power analog blocks put an A/D converter near 1 µA against roughly 150 nA for an amplifier stage; a six-layer network converting at every boundary carries ten more conversions than one converting only at its input and output. **This study models no power and claims none** — it bounds the accuracy side of that trade at ≤ 0.01 AUC, conditional on the noise budget above and on the training recipe below.

**The training recipe.** At σ_read = 0 all three `none` models are weak (0.597, 0.677, 0.675) with 3–6×
the chip-to-chip spread they show at 4 %. Evaluation batching (≤ 0.0008) and batch-norm statistics
(§4.2) each explain little of it. A cross-evaluation settles the question:

| Trained at σ_read (rows) · deployed at σ_read (columns) | 0 (quiet chip) | 4 % |
|---|---|---|
| 0 | 0.665 ± 0.042 | 0.657 ± 0.051 |
| 4 % | **0.738 ± 0.014** | 0.726 ± 0.012 |

**Activation noise during training, not the noise of the deployed chip, decides robustness.** A model trained with ≈ 0.24 LSB of activation noise is robust even on a perfectly quiet chip; one trained without it is fragile everywhere. `per_layer` does not need the step, plausibly because inter-layer quantisation supplies the same perturbation during training — the mechanism is not established. §3.4 reproduces the effect independently.

### 3.4 Bit width (Figure 4)

`none` placement trained with 4 % activation noise (A17); W / S / B varied jointly with QAT at each width; 3 seeds × 10 chips. *Nominal* matches training (σ_prog 3 %, σ_d2d 2 %, σ_read 4 %); *quantisation only* disables all noise.

![Figure 4](constraint-study/results/figures/fig4_bits.png)

| Bits | 2 | 4 | **6 (chip)** | 8 | 10 |
|---|---|---|---|---|---|
| Nominal | 0.538 ± 0.073 | 0.449 ± 0.005 | **0.729 ± 0.012** | 0.727 ± 0.013 | 0.727 ± 0.012 |
| Quantisation only | 0.593 ± 0.074 | 0.452 ± 0.001 | **0.738 ± 0.012** | 0.739 ± 0.012 | 0.740 ± 0.005 |

- **6 bits is sufficient and 4 bits fails.** 8 and 10 bits differ from 6 by −0.002 and +0.001; more
  resolution on W, S or B buys nothing on this task. At 4 bits the model collapses below chance with near-identical results across seeds (std ≤ 0.005) — a systematic degenerate solution that reconstructs anomalous clips *better* than normal ones. The cliff is somewhere between 4 and 6 bits; 5 bits was not tested, so the margin of the 6-bit specification is unknown.
- **Independent reproduction of the training recipe.** An earlier run of this sweep, trained without activation noise, gave 0.654 / 0.642 / 0.664 at 6 / 8 / 10 bits — 0.06–0.085 lower, with about 4× the spread.

---

## 4. Handoff and per-chip calibration

### 4.1 What the ML side delivers

For the target model the artefact is small enough to print:

| Item | Content |
|---|---|
| Tile map | 6 tiles, one per layer; no tile shared, no partial sums |
| Weight and bias codes | 6-bit signed integers, 4,242 weights |
| Tile scales | one real-valued scale per tile — a property of the tile's conductance range, not of the layer |
| BN gain and offset | one pair per channel, real-valued; assumed programmable (A5) |
| D/A and A/D full scale | input and output ranges. Currently a dynamic per-batch scale; a **fixed calibrated range** is required before this is a specification (A16) |
| Decision threshold | operating point on the reconstruction-MSE score, set from normal data at the target false-positive rate |
| Training conditions | the σ_prog, σ_d2d and activation-noise levels the model was trained under |

The last row is the one easily left out of a weights file and expensive to rediscover: a model is only valid for chips whose non-idealities are no worse than those it was trained against, and without those numbers a changed process cannot be checked against an existing model.

### 4.2 Whether per-chip calibration is worth its test time

Batch normalisation is the natural calibration handle. Re-estimating its statistics per chip on normal audio was measured directly (`scripts/probe_bn_calib.py`, 3 seeds × 10 chips); window counts are given with the equivalent length of a continuous recording:

| Model | No calibration | 2,048 windows (≈ 1 min) | 20,480 windows (≈ 11 min) |
|---|---|---|---|
| `none`, trained with 4 % activation noise | **0.731 ± 0.013** | 0.728 ± 0.017 | 0.731 ± 0.016 |
| `none`, trained without it | 0.648 ± 0.056 | 0.649 ± 0.050 | 0.670 ± 0.042 |
| `per_layer`, σ_read 0 | 0.732 ± 0.017 | 0.727 ± 0.016 | 0.731 ± 0.017 |

- **With the right training recipe, per-chip calibration buys nothing** (≤ 0.003 either way) — a per-chip step and its audio-capture time can come out of the production flow.
- **Calibration is not a substitute for the recipe.** For the fragile model, eleven minutes of audio per chip recovers 0.022 of a 0.083 gap and leaves its worst chip at 0.588.

---

## 5. Limitations

- **Simulation only.** Noise magnitudes come from the literature or are stress ranges (σ_read); none is measured on the target process (A2, A15). σ_read models additive noise alone — offset and gain error of the signal path are not represented (A18).
- **Not modelled:** retention drift (EEPROM, ReRAM and PCM differ fundamentally), temperature, rail saturation, and a saturating hidden non-linearity — the simulation uses ReLU (A4).
- **Quantisation scale.** Activation converters use a dynamic per-batch scale rather than a fixed calibrated range, so results depend on evaluation batch composition by up to ≈ 0.01 AUC for placements with many conversion points (≤ 0.0008 for `none`) (A16). No finding with a gap ≥ 0.03 is affected.
- **Statistics.** 3–4 seeds per condition; gaps below ≈ 0.03 are not claimed unless the paired differences are tight. Activation-noise-trained models evaluate ≈ 0.02 above the fp32 baseline, which at this sample size is not established; a dedicated fp32 + noise-injection control would settle it. HWA training occasionally collapses (1 of 24 and 1 of 54 runs), removed by the validation-loss filter. The mechanisms behind the training-recipe effect and the 4-bit collapse are both unexplained.
- **Not covered:** the 80 × 80 binary array; a learnable analog front end (features are digital log-mel); machine types other than ToyCar (MIMII would be closer to industrial equipment);
  biomedical signals; multi-tile cascading, deferred because its cost depends on whether read noise is absolute or signal-proportional; power and energy.

---

## 6. Open questions for the hardware team

These would change the conclusions most; the full list is in
[assumptions.md](assumptions.md#e-open-questions-for-the-hardware-team).

1. Are the taped-out array size and precision still 30 × 30 and 6 bits?
2. What weight-error distribution does the program-verify tolerance produce?
3. What are the noise, offset and gain error of the analog path between hidden layers — and is read
   noise absolute or proportional to the signal? The latter decides whether multi-tile cascading is free.
4. What is the actual hidden-layer non-linearity, and where does it saturate?
5. Is multi-tile cascading supported, and how are partial sums combined?
6. What activation rate corresponds to a given signal-path noise figure — the missing link between the
   accuracy budget in §3.3 and the throughput specification?
