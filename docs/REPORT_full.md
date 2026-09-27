# Feasibility of a 30 × 30, 6-bit Analog MAC Array for Machine Acoustic Anomaly Detection

*A simulation study · Kaiwen Lin · KaiwenLin@utexas.edu*

*Full report. The [README](../README.md) is the short version; the
[interactive page](https://catarinaseigward.github.io/analog-mac-anomaly-study/site/) plots every simulated chip.*

> **Simulation only; nothing here has been verified on silicon.** Hardware parameters are inferred from
> published test-chip specifications for analog in-memory inference or taken from the literature; each is
> listed with its status in [assumptions.md](../assumptions.md). The code and aggregated results behind every
> number are in [`constraint-study/`](../constraint-study/).

---

## Summary

The MLPerf Tiny anomaly-detection reference model, ten fully connected layers
`640 → [128]×4 → 8 → [128]×4 → 640`, occupies **380 tiles** of a 30 × 30 crossbar; one layer alone needs 110.
This study compresses it to six layers, `30 → [30]×2 → 4 → [30]×2 → 30`, that occupy **6 tiles**, one per layer,
and evaluates it on a behavioural model of a 6-bit analog MAC array with no A/D or D/A conversion between
hidden layers. Task: ToyCar machine-sound anomaly detection (DCASE 2020 Task 2). Every number is a mean over
3–4 training seeds × 10 simulated chips.

**It fits.** On the simulated array (6-bit W/S/B, 3 % programming error, 2 % device mismatch, signal-path noise,
no inter-layer A/D) the compressed model reaches **AUC 0.722 ± 0.012**, level with the same network in fp32
(0.719–0.727) and 78 % of the above-chance discrimination of the 380-tile reference (0.785 ± 0.008). The loss
comes from shrinking the model, not from the analog hardware. Under every revised device model it stays at
0.70–0.73. (§2.2, §3.1, §3.5)

Three findings matter most for the chip:

1. **Removing inter-layer A/D holds up; the noise of the signal path sets the limit.** With fixed converter
   ranges, as real converters have, removing inter-layer A/D costs nothing measurable (+0.007 ± 0.008 AUC).
   At ≈ 1 LSB of noise both placements fail together. The noise budget depends on the noise model: ≈ 0.5 LSB
   for signal-proportional noise, and a 40 dB column dynamic range still suffices under a fixed floor. (§3.3, §3.5)
2. **Hardware-aware training decides yield.** With it, AUC holds at 0.714–0.735 up to 3 % programming error, so a
   program-verify tolerance of ≈ 3 % is enough. Without it, some chips are at or below chance at every setting (8 of
   40 at 3 %), and the chip-to-chip spread is up to 5.9× wider. (§3.2)
3. **The converters set the bit-width floor, not the weights.** Weights and biases hold at 4 bits (−0.010 and
   +0.005). The two converters need 5 bits, or 4 if a per-channel offset is subtracted ahead of the D/A: a
   log-mel feature is mostly DC, and removing it shrinks the converter full scale 3.6×. With the offset, 4-bit
   W/S/B matches 6 bits (0.723 ± 0.016). (§3.4)

Three further results:

- **Spend the input budget on time, not frequency.** 6 mel bands × 5 frames beats 30 bands × 1 frame by
  +0.068 AUC (t = 4.2). Leaky integrators with τ ≤ 128 ms supply that context as well as stored frames. (§3.1)
- **Activation noise during training is a regulariser, and it belongs in the specification.** Without it the
  no-inter-layer-A/D model loses 0.087 AUC even on a quiet chip; per-chip calibration does not recover it, and
  with it calibration is unnecessary. (§3.3, §4.2)
- **The decision threshold must be measured on a chip.** A threshold from simulation flags every clip; one set
  on a reference chip transfers to the others. (§4.3)

---

## 1. Target and task

| Constraint | Value |
|---|---|
| Weight array | 30 × 30 |
| W / S / B precision | 6 bit each |
| A/D, D/A in hidden layers | none |
| Weight storage | in memory (EEPROM or ReRAM); fixed at inference |
| Operators | fully connected, batch normalisation, scalar multiply; no convolution |
| Operation | asynchronous, clockless, current mode |

The absence of convolution decides the model class; the array size decides everything else. Both are
parameters throughout the code (A1). **Task:** unsupervised machine-condition monitoring from sound, trained on
normal clips only and scored by reconstruction error, on the ToyCar machine type of DCASE 2020 Task 2 as used by
MLPerf Tiny.

---

## 2. Method

### 2.1 Reference model, data and protocol

The starting point is the MLPerf Tiny reference: a fully connected autoencoder over log-mel features
(128 bands, 5-frame window, n_fft 1024, hop 512), with a clip scored by the mean reconstruction MSE over its
windows. One model is trained on machine IDs 01–07 (7,000 normal clips) and evaluated on IDs 01–04 (1,400
normal, 1,059 anomalous). Reproduced here at AUC 0.7804 / pAUC 0.6737 against the published 0.8009 / 0.6722
(A9).

**Protocol.** AUC is the primary metric; pAUC (FPR ≤ 0.1) is recorded and follows the same trends. Every number
is a mean ± std over training seeds × 10 simulated chips, 3 seeds per condition (4 in §3.2); single seeds proved
unreliable (seed-to-seed std up to 0.065). Comparisons between conditions are paired by seed. A run whose final
*validation* loss exceeds twice the median of comparable runs is excluded; test AUC never enters that filter,
and no run in this report was excluded. Sweeps use a faster training configuration, accepted after it matched the
reference on both metrics (ΔAUC −0.003, ΔpAUC −0.008; A8). All values are in `constraint-study/results/*.csv`.

### 2.2 Mapping the network onto tiles

A weight matrix `[out, in]` occupies `ceil(out / 30) × ceil(in / 30)` tiles, but the two directions are not
equally cheap:

- **Output-direction tiling** splits a layer across tiles computing disjoint outputs. They do not interact; the
  cost is area and programming time.
- **Input-direction tiling** splits a dot product into partial sums added on a wire (Kirchhoff's current law).
  Every contributing tile adds its own mismatch and noise to the same node. **Input-direction tiling is where
  accuracy is spent.**

| Stage | Reference (10 layers) | Tiles | Target (6 layers) | Tiles |
|---|---|---|---|---|
| Input layer | FC 640 → 128 | **110** (22 × 5) | FC 30 → 30 | 1 |
| Encoder blocks | 3 × FC 128 → 128 | 75 | 1 × FC 30 → 30 | 1 |
| Into bottleneck | FC 128 → 8 | 5 | FC 30 → 4 | 1 |
| Out of bottleneck | FC 8 → 128 | 5 | FC 4 → 30 | 1 |
| Decoder blocks | 3 × FC 128 → 128 | 75 | 1 × FC 30 → 30 | 1 |
| Output layer | FC 128 → 640 | **110** (5 × 22) | FC 30 → 30 | 1 |
| **Total** | | **380** | | **6** |
| Parameters | | 267,928 | | 4,242 |

The reference's first layer puts 22 partial-sum currents on every output node. The target tiles in neither
direction, so multi-tile cascading, the one capability this study could not confirm (A7), is not required.

### 2.3 Behavioural model of the array

Each fully connected layer is replaced by a simulated analog layer (`src/models/analog.py`):

- **Quantisation** of weights, signals and biases at 6 bits, with one weight scale per 30 × 30 tile. The
  straight-through estimator passes the gradient unless a value is clipped.
- **Device-to-device mismatch** (σ_d2d = 2 %): multiplicative per weight, drawn once per chip and frozen; redrawn
  every training step so the model cannot specialise to one chip. Chips come from a random stream independent of
  the training seeds.
- **Programming / cycle-to-cycle error** (σ_prog): multiplicative per weight, redrawn every inference.
- **Signal-path read noise** (σ_read): additive Gaussian on each layer's output, std = σ_read · mean|W·x|. At
  6 bits, 4 / 8 / 17 / 34 % correspond to ≈ 0.24 / 0.47 / 1.0 / 2.0 LSB. A **stress range, not a device
  parameter** (A15).
- **Converter placement:** `none` quantises only the first layer's input and the last layer's output
  (2 conversions); `per_layer` quantises the input and output of every layer (12).
- **Hardware-aware training (HWA):** the network is trained with the analog layers in place, then deployed to
  10 simulated chips without retraining.

Batch normalisation is a separate affine stage, assumed to be a programmable gain and offset (A5); the hidden
non-linearity is ReLU (A4).

**Revised device model (§3.5).** Three switchable changes:

- **Fixed converter ranges:** each D/A and A/D full scale is calibrated in training (moving average of the
  99.9th percentile of |signal|) and frozen at evaluation, instead of a max-abs scale recomputed per batch (A16).
- **Saturating non-linearity:** a clipped ReLU `min(max(z, 0), 1)`, the shape of a unipolar current mirror
  limited by its bias, or tanh, the shape of a differential pair.
- **Read-noise models in hardware units**, with U the cell full-scale current: *absolute* (std = σ · U;
  column dynamic range 20·log10(30 / σ)), *proportional* (std = σ · |W·x| per element) and *shot*
  (std = σ · √(U · |W·x|)). Because U is fixed by hardware, a network cannot escape absolute or shot noise by
  shrinking its signals.

---

## 3. Results

### 3.1 What fits (Figure 1)

fp32 training, 3 seeds.

![Figure 1](../constraint-study/results/figures/fig1_dimension.png)

| | Input | Tiles (largest layer) | Parameters | AUC |
|---|---|---|---|---|
| Reference | 128 bands × 5 frames = 640 | 380 (110) | 267,928 | 0.785 ± 0.008 |
| **Target** | 6 bands × 5 frames = 30 | **6 (1)** | **4,242** | 0.727 ± 0.012 |

The target keeps 80 % of the reference's above-chance discrimination with 63× fewer tiles and parameters.
Frequency resolution costs about 0.021 AUC per halving (640 / 320 / 160 / 80 dimensions: 0.785 / 0.762 / 0.742 /
0.721).

**How to spend 30 input lines** (hidden width 30):

| Bands × frames | 6 × 5 | 10 × 3 | 16 × 2 | 30 × 1 | 32 × 1 |
|---|---|---|---|---|---|
| AUC | **0.727 ± 0.012** | 0.717 ± 0.011 | 0.714 ± 0.015 | 0.659 ± 0.026 | 0.662 ± 0.022 |

Every configuration with at least two frames scores ≥ 0.713; both single-frame ones score ≤ 0.662
(6 × 5 against 30 × 1: +0.068, t = 4.2). The 30 input lines are the scarce resource, and they are better spent on
a few bands with several time samples than on a finely resolved instantaneous spectrum.

**Leaky integrators instead of stored frames.** Replacing the 5 frames by 5 first-order low-pass outputs per
band, read at the same instant (both arms at a 4× lower learning rate, which the collinear features need):

| Features (6 bands × 5 channels) | fp32 | Simulated chip |
|---|---|---|
| 5 stacked frames | 0.730 ± 0.019 | 0.716 ± 0.005 |
| Leaky integrators, τ = 32–128 ms | **0.751 ± 0.014** (paired +0.021) | 0.703 ± 0.019 (paired −0.013 ± 0.016) |
| Leaky integrators, τ = 32–512 ms | 0.708 ± 0.017 (paired −0.022) | — |

Time constants up to about 128 ms match stored frames; longer ones hurt. Short time constants are the cheap
ones in a design without passive capacitors.

### 3.2 Programming error and hardware-aware training (Figure 2)

σ_d2d = 2 %, 4 seeds × 10 chips. **C1** fp32 training and inference; **C2** fp32 training, analog inference;
**C3** HWA; **C4** HWA plus 6-bit quantisation at every layer output.

![Figure 2](../constraint-study/results/figures/fig2_noise_stefix.png)

| σ_prog | C2 (no HWA) | C3 (HWA) | Worst chip, C2 | Worst chip, C3 | Chips below chance, C2 |
|---|---|---|---|---|---|
| 0 % | 0.689 ± 0.069 | **0.735 ± 0.023** | 0.497 | 0.681 | 1 of 40 |
| 1 % | 0.669 ± 0.083 | **0.716 ± 0.014** | 0.431 | 0.688 | 3 of 40 |
| 2 % | 0.643 ± 0.086 | **0.715 ± 0.020** | 0.379 | 0.682 | 2 of 40 |
| 3 % | 0.620 ± 0.108 | **0.714 ± 0.024** | 0.311 | 0.654 | 8 of 40 |
| 5 % | 0.608 ± 0.078 | **0.701 ± 0.025** | 0.430 | 0.640 | 5 of 40 |
| 10 % | 0.542 ± 0.072 | **0.670 ± 0.050** | 0.373 | 0.584 | 10 of 40 |

C1 = 0.720 ± 0.018; C4 − C3 stays within −0.011 … 0.000. No C3 chip falls below chance at any setting.

- **Up to 3 % programming error is nearly free with HWA** (0.714–0.735). Relative to 3 %, 5 % costs 0.013 and
  10 % costs 0.044. Tighter programming buys no accuracy but costs programming time, which scales with weight
  count and test cost.
- **The mean is the wrong number to read.** Without HWA the mean looks survivable at 0.62–0.69 while individual
  chips fall below chance; a part that ships at 0.43 is a returned part. HWA narrows the spread up to 5.9×.

These runs quantise the input of every layer; removing inter-layer conversion is tested in §3.3, and at the
nominal operating point the two agree (0.714 ± 0.024 against 0.722 ± 0.012).

### 3.3 Inter-layer A/D and signal-path noise (Figure 3)

Both placements, each trained with HWA under its own conditions; σ_prog 3 %, σ_d2d 2 %, 3 seeds × 10 chips,
paired by seed.

![Figure 3](../constraint-study/results/figures/fig3_adc_stefix.png)

| σ_read | ≈ noise / LSB | `none` | `per_layer` | Paired difference |
|---|---|---|---|---|
| 0 | 0 | 0.663 ± 0.077 | 0.728 ± 0.010 | −0.065 ± 0.052 (training recipe, below) |
| 4 % | 0.24 | 0.723 ± 0.011 | 0.733 ± 0.014 | **−0.010 ± 0.019** |
| 8 % | 0.47 | 0.718 ± 0.012 | 0.730 ± 0.008 | **−0.012 ± 0.012** |
| 17 % | 1.0 | 0.574 ± 0.006 | 0.575 ± 0.009 | −0.001 ± 0.005 |
| 34 % | 2.0 | 0.495 ± 0.008 | 0.494 ± 0.010 | +0.001 ± 0.004 |

Under this device model, removing the inter-layer converters costs about 0.01 AUC below ≈ 0.5 LSB, growing with
depth (−0.007 / −0.012 / −0.027 at 1 / 2 / 4 blocks, confounded with parameter count). At ≈ 1 LSB both placements
fail together, and the transition is steep: between 0.47 and 1.0 LSB, AUC falls from 0.72 to 0.57. **What sets the
limit is the noise of the signal path, not the placement of the converters.**

Two qualifications come from §3.5. With fixed converter ranges the cost of removing inter-layer A/D is within
noise (+0.007 ± 0.008): part of `per_layer`'s advantage here comes from its twelve converters re-ranging every
batch, which real converters cannot do. And the 0.5-LSB budget is specific to signal-proportional noise.

Published figures for ultra-low-power analog blocks put an A/D converter near 1 µA against about 150 nA for an
amplifier stage, and `per_layer` carries ten more conversions. **This study models no power**; it bounds the
accuracy side of that trade.

**The training recipe.** At σ_read = 0 the `none` models are weak, with 4–9× the chip-to-chip spread they show
at 4 %. A cross-evaluation locates the cause:

| Trained at σ_read · deployed at σ_read | 0 (quiet chip) | 4 % |
|---|---|---|
| 0 | 0.646 ± 0.088 | 0.648 ± 0.071 |
| 4 % | **0.733 ± 0.014** | 0.720 ± 0.013 |

**Activation noise during training, not the noise of the deployed chip, decides robustness.** The mechanism is
regularisation: in plain fp32, with no quantisation or device noise, injecting activation noise during training
raises AUC by +0.014 ± 0.009 at 4 % and +0.027 ± 0.016 at 8 %, while noise on the input does not help. The
gain still rises at 8 %, so 4 % is not necessarily the best recipe. `per_layer` does not need the step,
plausibly because its inter-layer quantisation supplies a similar perturbation.

### 3.4 Bit width (Figure 4)

`none` placement trained with 4 % activation noise; quantisation-aware training at each width; 3 seeds ×
10 chips. *Nominal* matches training (σ_prog 3 %, σ_d2d 2 %, σ_read 4 %).

![Figure 4](../constraint-study/results/figures/fig4_bits_v2.png)

All three together, nominal noise: 0.540 ± 0.027 at 4 bits, 0.710 ± 0.018 at 5, **0.722 ± 0.012** at 6,
0.726 ± 0.011 at 8 and 0.730 ± 0.016 at 10. **One at a time**, the other two at 6 bits:

| Bits | W (conductance levels) | S (input D/A + output A/D) | S, per-channel offset ahead of the D/A | B (bias) |
|---|---|---|---|---|
| 4 | **0.712 ± 0.015** | 0.570 ± 0.023 | 0.762 ± 0.018 | **0.727 ± 0.017** |
| 5 | 0.708 ± 0.014 | 0.721 ± 0.021 | 0.726 ± 0.011 | 0.724 ± 0.013 |
| 6 | 0.722 ± 0.012 | ← | 0.726 ± 0.009 | ← |

- **Weights and biases hold at 4 bits** (−0.010 and +0.005). More than 6 bits buys nothing.
- **The two converters set the floor.** A log-mel feature is a large DC level (about −26 dB, from −41 to −10
  across channels) with a small fluctuation (about 3.5 dB); a converter spends its range on the DC, and its
  4-bit step (8.4 dB) is larger than the signal it carries.
- **A per-channel offset ahead of the D/A** (the training-set mean of each input line, added back after the
  output A/D) shrinks the converter full scale 3.6×, from 52.3 to 14.5 dB, or about 1.85 bits. With it, 4-bit
  converters cost nothing. Under fixed converter ranges (§3.5), all three widths at 4 bits give
  **0.723 ± 0.016**, paired +0.000 ± 0.015 against 6 bits; without the offset, 0.684 ± 0.019.
- The excess of the offset column over 6 bits is not claimed: a coarser output A/D reshapes the anomaly score,
  and the same change applied at evaluation raises the 6-bit model by about 0.02.
- 2 bits is not a working model: the input carries two levels and the output is constant.

*Correction.* An earlier version of this report found that 4 bits fails below chance. That result came from a
bug in the straight-through estimator, which gave no gradient to values on the top quantisation level; it has
been fixed, covered by regression tests, and every quantised result here was retrained on the fixed code.

### 3.5 Robustness to the device model

Target configuration, 3 seeds × 10 chips, each model trained under the device model it is evaluated on;
σ_prog 3 %, σ_d2d 2 %.

| Device model | `none` | `per_layer` | `none` − `per_layer` |
|---|---|---|---|
| Fixed converter ranges | 0.724 ± 0.017 | 0.717 ± 0.010 | **+0.007 ± 0.008** |
| + clipped ReLU | 0.724 ± 0.025 | 0.720 ± 0.008 | +0.004 ± 0.022 |
| + absolute noise σ 0.03 (60 dB) | 0.719 ± 0.010 | 0.721 ± 0.008 | −0.003 ± 0.002 |
| + shot noise σ 0.02 | 0.710 ± 0.023 | 0.718 ± 0.018 | −0.008 ± 0.026 |
| + proportional noise 4 % | 0.702 ± 0.018 | 0.718 ± 0.010 | **−0.016 ± 0.006** |
| tanh instead of clipped ReLU, absolute σ 0.03 | 0.727 ± 0.015 | — | — |

- **Fixed converter ranges** leave `none` unchanged and cost `per_layer` 0.017: with physical converters,
  removing inter-layer A/D costs nothing measurable.
- **The shape of the non-linearity does not matter** here (clipped ReLU against ReLU: within 0.003; tanh
  against clipped ReLU: +0.008).
- **Only per-element proportional noise costs accuracy** (about 0.02), and only under it does removing
  inter-layer A/D have a measurable cost (−0.016 ± 0.006). Which model the real read noise follows is open
  question 3 in §6.

**Dynamic range.** Absolute noise, fixed converter ranges, clipped ReLU; σ in units of U:

| σ | Column dynamic range | `none` | `per_layer` |
|---|---|---|---|
| 0.01 | 70 dB | 0.705 ± 0.038 | 0.726 ± 0.014 |
| 0.03 | 60 dB | 0.719 ± 0.010 | 0.721 ± 0.008 |
| 0.1 | 50 dB | 0.726 ± 0.021 | 0.703 ± 0.010 |
| 0.3 | 40 dB | 0.715 ± 0.041 | 0.737 ± 0.008 |

No transition appears between 70 and 40 dB, because the network adapts: the ratio of mean signal to U near the
bottleneck grows from about 0.2 at σ 0.01 to 0.9 at σ 0.3, and more hidden activations sit on the rail. The
network spends its headroom to hold its signal-to-noise ratio, which the signal-proportional model of §3.3
cannot show. **40 dB is enough on this task; the actual limit is lower and was not reached**, though at 40 dB
one `none` seed already drops to 0.659.

---

## 4. Handoff, calibration and the decision threshold

### 4.1 What the ML side delivers

| Item | Content |
|---|---|
| Tile map | 6 tiles, one per layer; no tile shared, no partial sums |
| Weight and bias codes | signed integers, 4,242 parameters; 6 bits specified, 4 bits sufficient (§3.4) |
| Tile scales | one scale per tile, a property of the tile's conductance range |
| BN gain and offset | one pair per channel; assumed programmable (A5) |
| Input offset | one value per input line, subtracted ahead of the D/A and added back after the output A/D; only for 4-bit converters (§3.4) |
| D/A and A/D full scale | fixed, at the 99.9th percentile of the training signal: 52.3 dB (D/A) and 52.3–53.4 dB (A/D) in log-mel units; with the input offset, ±14.5 dB and ±13.6 dB around each channel's mean; 0.12–0.14 % of test samples clip |
| Decision threshold | measured on a reference chip from normal audio (§4.3) |
| Training conditions | the σ_prog, σ_d2d and activation-noise levels the model was trained under |

The last row is the one easily left out of a weights file: a model is only valid for chips whose
non-idealities are no worse than those it was trained against.

### 4.2 Whether per-chip calibration is worth its test time

Re-estimating batch-normalisation statistics per chip on normal audio, 3 seeds × 10 chips:

| Model | No calibration | ≈ 1 min of audio | ≈ 11 min of audio |
|---|---|---|---|
| `none`, trained with 4 % activation noise | **0.720 ± 0.011** | 0.720 ± 0.016 | 0.726 ± 0.009 |
| `none`, trained without it | 0.643 ± 0.078 | 0.641 ± 0.071 | 0.661 ± 0.065 |
| `per_layer`, σ_read 0 | 0.732 ± 0.010 | 0.730 ± 0.013 | 0.732 ± 0.010 |

With the right training recipe, per-chip calibration buys little (+0.001 to +0.006), so the step and its
audio-capture time can leave the production flow. It is no substitute for the recipe: for the fragile model,
eleven minutes of audio recovers 0.018 of a 0.089 gap.

### 4.3 Setting the decision threshold

Threshold set per machine at a 10 % false-positive target from half of the normal test clips (standing in for
audio recorded at installation) and evaluated on the rest; 10 chips × 3 seeds, nominal noise.

| Threshold set on | 6-bit model: false-positive rate | 4-bit with input offset |
|---|---|---|
| The noise-free model (simulation) | **100 %** | 8–16 % |
| One reference chip, applied to the other nine | 8–13 % | 6–11 % |
| Each chip, at installation | 8–11 % | 7–9 % |

Device noise raises the scores of normal clips 1.80× in the 6-bit model, so a threshold computed without it flags
everything; the input offset shrinks that shift to 1.05×. **Measure the threshold on a reference chip**; per-chip
thresholds add little. The operating point on this benchmark is modest: at a 10 % false-positive rate the detector
finds about 37 % of anomalies, and precision is about 0.30 if one clip in ten is anomalous. These absolute numbers
belong to ToyCar at AUC ≈ 0.72; the relative costs above are what transfer.

---

## 5. Limitations

- **Simulation only.** Noise magnitudes come from the literature or are stress ranges; none is measured on the
  target process (A2, A15). The three read-noise models test whether the conclusions depend on the noise model,
  not which one the silicon follows.
- **Not modelled:** retention drift, temperature, offset and gain error of the signal path between layers (A18),
  power, and a front end simulated from the raw waveform.
- **Statistics.** 3 seeds per condition (4 in §3.2); differences within about ±0.02 in §3.5 are not significant,
  and the dynamic-range limit was not located. The programming-error and depth sweeps were not repeated under the
  revised device model.
- **One benchmark.** On ToyCar a constant output already reaches AUC 0.52–0.82 depending on the constant, so
  MIMII would be a stronger test. Multi-tile cascading and the 80 × 80 binary array were not studied.

---

## 6. Open questions for the hardware team

The full list is in [assumptions.md](../assumptions.md#e-open-questions-for-the-hardware-team).

1. Are the taped-out array size and precision still 30 × 30 and 6 bits?
2. What weight-error distribution does the program-verify tolerance produce?
3. What are the noise, offset and gain error of the analog path between hidden layers, and is read noise a
   fixed floor or proportional to the signal? It decides the noise budget and whether removing inter-layer A/D
   is free (§3.5).
4. What is the hidden-layer non-linearity, and where does it saturate? Its shape does not matter here; the
   dynamic range between rail and noise floor does.
5. Is multi-tile cascading supported, and how are partial sums combined?
6. Is a per-channel offset ahead of the D/A cheap in the front end? It decides whether the converters can drop
   from 6 to 4 bits (§3.4).
