# Assumptions

Last reviewed: 23 September 2026, for version 2 of the [full report](docs/REPORT_full.md) — after every
quantized result was retrained on the corrected code and re-tested under a revised device model. "Report §…"
below refers to the full report.

This is a **simulation study**; nothing has been verified on silicon. Target-hardware parameters are
inferred from publicly available material or taken from the literature. Every assumption is listed
below with its basis and status. Items marked ⚠️ need confirmation from the hardware team before any
recommendation in the report is acted on.

Identifiers are kept stable across revisions, so they are not sequential within sections.

---

## A. Target hardware

| # | Assumption | Basis | Status |
|---|---|---|---|
| **A1** | Weight array 30 × 30; W / S / B 6-bit each | Public NN-IC test-chip specification (TSMC 180 nm; silicon availability listed as 2Q-2025) | ⚠️ Public pages have changed since this study began (they previously carried a *"Work in Progress"* marker dated 4Q24), so the taped-out part may differ again. Array size and bit widths are parameters in the code (`DeviceParams`). |
| **A2** | Weight-programming error is multiplicative Gaussian, σ_prog ∈ [0, 10 %] | Program-verify tolerance band; range from PCM / ReRAM literature | ⚠️ Not measured on the target process. |
| **A3** | No A/D or D/A conversion between hidden layers | Public statement: *"eliminates the need for external memory and costly A/D or D/A conversions in hidden layers"* | Derived. Simulated as `adc_mode="none"` (report §3.3–3.4). Figure 2 (§3.2) used an earlier placement that quantises every layer input; at the nominal operating point the two agree (0.718 vs 0.729). |
| **A4** | Hidden-layer non-linearity | — | **Tested (report §3.5).** ReLU in the main results; a clipped ReLU (unipolar current mirror limited by its bias) and tanh (differential pair) change AUC by +0.0005 to +0.008 — the shape does not matter on this task. The rail position is absorbed by the learnable batch-norm gain; what matters is the dynamic range between rail and noise floor. The actual non-linearity still needs confirming. |
| **A5** | Batch normalisation is available at inference as a programmable per-channel gain and offset, usable for per-chip calibration | Public statement that batch normalisation is supported | Derived. Per-chip re-estimation of its statistics was tested (report §4.2): it recovers ≈ 0.02 AUC in the one fragile case and changes nothing otherwise. |
| **A6** | Device-to-device mismatch is fixed per chip (σ_d2d = 2 %); programming / cycle-to-cycle error is redrawn on every inference | Standard analog in-memory-compute modelling convention | Convention |
| **A7** | Layers larger than 30 × 30 would be tiled, with input-direction partial sums added as currents | Standard crossbar practice | ⚠️ Multi-tile cascading is unconfirmed. The target model does not need it: every layer fits a single tile. |
| **A19** | Six 30 × 30 arrays are available on one die, and a layer's analog output can drive the next layer's input array without conversion | Required by `adc_mode="none"`: the architecture only exists if hidden-layer signals reach the next array in the analog domain | ⚠️ **Unconfirmed, and the most load-bearing unlisted assumption.** The public specification describes one 30 × 30 array and does not say how many are on a die. The target model was sized so that no *layer* needs tiling (A7), but six layers still have to be reachable in sequence. With a single array the layers would have to be time-multiplexed, which conflicts with weights-in-memory. |
| **A15** | Signal-path read noise is additive Gaussian, std = σ_read · mean\|W·x\| per layer, swept over 0–34 % (≈ 0–2 LSB of a 6-bit A/D) | Modelling choice: a stress range that makes A/D noise regeneration observable | ⚠️ **Not a device parameter.** Three alternative models were tested (report §3.5), defined in hardware units: an absolute floor, per-element proportional, and shot noise. The conclusions hold under all three except that per-element proportional noise costs about 0.02 AUC and makes removing inter-layer A/D cost 0.016. Under an absolute floor the network uses its headroom; a 40 dB column dynamic range still suffices and the limit was not reached. The real specification — and which model applies — must come from the analog design team. |
| **A18** | σ_read is treated as the model-side image of whatever degrades the signal path as activations are driven faster (report §3.3, against the public claim that accuracy *"degrades smoothly with increasing activation signal frequency, even without S/H"*) | Interpretation | ⚠️ **No calibrated mapping** between an activation frequency and an LSB figure was established. Only additive noise is modelled; offset and gain error of the signal path are not. |

### Not modelled

- **Retention drift.** EEPROM (floating-gate charge loss), ReRAM (conductance relaxation) and PCM
  (power-law drift) differ fundamentally; the PCM power law common in the literature must not be
  applied to the other two. `DeviceParams.drift` exists but is 0 in all experiments.
- **Temperature** dependence of gains, time constants and filter frequencies.
- **Offset and gain error** of the analog signal path between layers — only additive noise is modelled (A18).
- **Power and energy.** The public nano-power figures (A/D ≲ 1 µA, smart amplifier ≲ 150 nA) are quoted in the report to motivate the converter-placement question; no power number is computed or claimed.
- **An analog front end simulated from the waveform.** Features are digital log-mel spectra. The front-end
  recommendation (Summary, finding 2) now rests on a leaky-integrator check (report §3.1), which integrates
  log-mel bands — a first-order approximation of rectify, integrate, then compress — not on a simulated
  filter bank.

*Saturation at the rails, previously listed here, is now modelled as a saturating hidden non-linearity
(A4, report §3.5).*

---

## B. Methodology

| # | Assumption | Basis | Status |
|---|---|---|---|
| **A8** | Sweeps use a fast training configuration (80 epochs, batch 2048, lr 0.002, window stride 2) instead of the reference one (100, 512, 0.001, 1) | Validated at the reference operating point: ΔAUC −0.0031, ΔpAUC −0.0077, both within 0.01; 5.6× faster | Validated |
| **A9** | The reproduced reference sits 0.0205 AUC below the published MLPerf Tiny value (0.7804 vs 0.8009) | Keras ↔ PyTorch differences: initialisation, batch-norm momentum convention, validation split. pAUC matches to +0.0015. | Known deviation, not chased |
| **A10** | Clip anomaly score = mean reconstruction MSE over **all** windows of the clip; evaluation uses every window | Reference implementation | Verified |
| **A11** | Train / test splits are by machine ID, never by recording | Dataset protocol | Verified |
| **A12** | 3–4 training seeds per condition × 10 simulated chips establish the reported effects | Seed-to-seed std up to 0.065 | ⚠️ Adequate for the reported gaps (≥ 0.03, or tight paired differences); differences within about ±0.02 in report §3.5 are not significant. The earlier open item — activation-noise-trained models evaluating ≈ 0.02 above fp32 — is **settled**: a plain-fp32 control shows +0.014 at 4 % and +0.027 at 8 % (paired, 3/3 seeds). |
| **A16** | Activation D/A and A/D quantisation uses a **dynamic per-batch max-abs scale**, not a fixed calibrated range | Implementation choice (version 1) | **Resolved in the revised device model (report §2.3, §3.5).** Converter full scales are calibrated at the 99.9th percentile during training and frozen. `none` is unchanged (+0.001); `per_layer` loses 0.017, so part of its version-1 advantage came from re-ranging twelve converters every batch. Main results keep the dynamic scale for continuity with version 1. |
| **A17** | Models of the no-inter-layer-A/D architecture are trained with 4 % activation noise, also when deployed on a quiet signal path | Cross-evaluation: training-time activation noise, not deployment-time noise, removes a −0.087 AUC fragility (report §3.3) | Result-driven choice; used for the bit-width sweep (report §3.4). Mechanism: regularisation — it raises AUC in plain fp32 too, and still rises at 8 %, so 4 % may not be optimal. |
| **A20** | A per-channel offset (the training-set mean of each input line) can be subtracted ahead of the D/A and added back after the output A/D | Proposed: a log-mel feature is mostly DC, so the converter range is otherwise spent on it | ⚠️ **Simulated only (report §3.4).** It shrinks the converter full scale 3.6× and lets the converters drop to 4 bits. Whether it is cheap in the front end is open question 12. |

---

## C. Data

| # | Assumption | Basis |
|---|---|---|
| **A13** | The Kaggle mirrors `daisukelab/dc2020task2` and `daisukelab/dc2020task2added` are faithful copies of the Zenodo originals | File counts match the official description: 4,000 + 3,000 training clips (machine IDs 01–07) and 2,459 test clips (IDs 01–04; 1,400 normal, 1,059 anomalous). Zenodo was unreachable from the development machine. |
| **A14** | ToyCar is a reasonable proxy for industrial acoustic condition monitoring | It is the MLPerf Tiny benchmark machine type. ⚠️ MIMII (valve, pump, fan, slide rail) would be closer to real industrial equipment. |

---

## D. Not covered

- The 80 × 80 binary (BNN) array.
- A learnable analog front end (filter bank, envelope detection, gain control).
- Heart-sound (PCG) data and machine types other than ToyCar.
- Multi-tile cascading — deferred in version 1, because under a signal-proportional noise model extra tiles
  cost nothing. The revised device model now has an absolute noise floor that adds per input tile, so the
  experiment is meaningful; it has not been run (question 9).
- Power and energy; real silicon.

*The 5-bit point and the fp32 + activation-noise control, previously listed here, are now in the report
(§3.4, §3.3).*

---

## E. Open questions for the hardware team

Answers to these would change the conclusions most directly.

1. Are the taped-out array size and bit widths still 30 × 30 and 6 bits? (A1)
2. What weight-error distribution does the program-verify tolerance band produce, and with what σ? (A2)
3. What is the actual hidden-layer non-linearity, and where does it saturate? (A4) On this task its shape does
   not matter; the dynamic range between the rail and the noise floor does.
4. How many 30 × 30 arrays are on a die, and can one layer's analog output drive the next layer's input
   array directly? (A19)
5. Is multi-tile cascading within a layer supported, and how are input-direction partial sums
   combined? (A7)
6. Is weight storage EEPROM or ReRAM, and which retention-drift model applies?
7. Are the batch-normalisation gain and offset programmable per chip, and what does calibration look like? (A5)
8. What are the measured magnitudes of device-to-device mismatch and cycle-to-cycle noise? (A6)
9. Is each tile's partial-sum read noise an absolute floor, independent of the signal, or proportional
   to the signal current? This decides whether multi-tile cascading has a cost, how large the signal-path
   noise budget is, and whether removing inter-layer A/D is free (report §3.5). (A7, A15)
10. What are the noise, offset and gain error of the analog signal path between hidden layers
   (current mirrors, transimpedance stages)? (A15)
11. What activation frequency corresponds to a given signal-path noise figure? This is the missing link
    between the accuracy budget in report §3.3 and the throughput specification. (A18)
12. Is a per-channel offset ahead of the D/A cheap in the front end? It decides whether the converters can
    drop from 6 to 4 bits. (A20)
