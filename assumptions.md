# Assumptions

Last reviewed: 12 September 2026, after all experiments in [REPORT.md](REPORT.md).

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
| **A3** | No A/D or D/A conversion between hidden layers | Public statement: *"eliminates the need for external memory and costly A/D or D/A conversions in hidden layers"* | Derived. Simulated as `adc_mode="none"` (report §3.4–3.5). Figure 2 (§3.3) used an earlier placement that quantises every layer input; at the nominal operating point the two agree (0.718 vs 0.729). |
| **A4** | Hidden-layer non-linearity | — | ⚠️ The simulation uses **ReLU**. A current-mode circuit is more likely to saturate (tanh-like); the effect of that difference is not studied. |
| **A5** | Batch normalisation is available at inference as a programmable per-channel gain and offset, usable for per-chip calibration | Public statement that batch normalisation is supported | Derived. Per-chip re-estimation of its statistics was tested (report §3.4): it recovers ≈ 0.02 AUC in the one fragile case and changes nothing otherwise. |
| **A6** | Device-to-device mismatch is fixed per chip (σ_d2d = 2 %); programming / cycle-to-cycle error is redrawn on every inference | Standard analog in-memory-compute modelling convention | Convention |
| **A7** | Layers larger than 30 × 30 would be tiled, with input-direction partial sums added as currents | Standard crossbar practice | ⚠️ Multi-tile cascading is unconfirmed. The target model does not need it: every layer fits a single tile. |
| **A15** | Signal-path read noise is additive Gaussian, std = σ_read · mean\|W·x\| per layer, swept over 0–34 % (≈ 0–2 LSB of a 6-bit A/D) | Modelling choice: a stress range that makes A/D noise regeneration observable | ⚠️ **Not a device parameter.** The real specification — and whether read noise is absolute or signal-proportional — must come from the analog design team. |
| **A18** | σ_read is treated as the model-side image of whatever degrades the signal path as activations are driven faster (report §3.4, against the public claim that accuracy *"degrades smoothly with increasing activation signal frequency, even without S/H"*) | Interpretation | ⚠️ **No calibrated mapping** between an activation frequency and an LSB figure was established. Only additive noise is modelled; offset and gain error of the signal path are not. |

### Not modelled

- **Retention drift.** EEPROM (floating-gate charge loss), ReRAM (conductance relaxation) and PCM
  (power-law drift) differ fundamentally; the PCM power law common in the literature must not be
  applied to the other two. `DeviceParams.drift` exists but is 0 in all experiments.
- **Temperature** dependence of gains, time constants and filter frequencies.
- **Offset and gain error** of the analog signal path between layers — only additive noise is modelled (A18).
- **Power and energy.** The public nano-power figures (A/D ≲ 1 µA, smart amplifier ≲ 150 nA) are quoted in the report to motivate the converter-placement question; no power number is computed or claimed.
- **Signal saturation** at the supply rails. `DeviceParams.clip` exists but is 0 in all experiments.
- **An analog front end.** Features are digital log-mel spectra. The front-end recommendation in the
  report (§4, item 2) rests on the time-versus-frequency result, not on a simulated filter bank.

---

## B. Methodology

| # | Assumption | Basis | Status |
|---|---|---|---|
| **A8** | Sweeps use a fast training configuration (80 epochs, batch 2048, lr 0.002, window stride 2) instead of the reference one (100, 512, 0.001, 1) | Validated at the reference operating point: ΔAUC −0.0031, ΔpAUC −0.0077, both within 0.01; 5.6× faster | Validated |
| **A9** | The reproduced reference sits 0.0205 AUC below the published MLPerf Tiny value (0.7804 vs 0.8009) | Keras ↔ PyTorch differences: initialisation, batch-norm momentum convention, validation split. pAUC matches to +0.0015. | Known deviation, not chased |
| **A10** | Clip anomaly score = mean reconstruction MSE over **all** windows of the clip; evaluation uses every window | Reference implementation | Verified |
| **A11** | Train / test splits are by machine ID, never by recording | Dataset protocol | Verified |
| **A12** | 3–4 training seeds per condition × 10 simulated chips establish the reported effects | Seed-to-seed std up to 0.065 | ⚠️ Adequate for the reported gaps (≥ 0.03, or tight paired differences). Smaller gaps are not claimed — for example, activation-noise-trained models evaluating ≈ 0.02 above fp32. |
| **A16** | Activation D/A and A/D quantisation uses a **dynamic per-batch max-abs scale**, not a fixed calibrated range | Implementation choice | ⚠️ Evaluation results depend on batch composition by up to ≈ 0.01 AUC for placements with many conversion points (≤ 0.0008 for `none`). A fixed calibrated range is the intended fix. |
| **A17** | Models of the no-inter-layer-A/D architecture are trained with 4 % activation noise, also when deployed on a quiet signal path | Cross-evaluation: training-time activation noise, not deployment-time noise, removes a −0.08 AUC fragility (report §3.4) | Result-driven choice; used for the bit-width sweep (report §3.5). |

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
- Multi-tile cascading — deferred, because under the current noise model extra tiles cost nothing and
  the outcome would be decided by the modelling choice rather than by the experiment (question 8).
- The 5-bit point, and an fp32 + activation-noise control.
- Power and energy; real silicon.

---

## E. Open questions for the hardware team

Answers to these would change the conclusions most directly.

1. Are the taped-out array size and bit widths still 30 × 30 and 6 bits? (A1)
2. What weight-error distribution does the program-verify tolerance band produce, and with what σ? (A2)
3. What is the actual hidden-layer non-linearity, and where does it saturate? (A4)
4. Is multi-tile cascading supported, and how are input-direction partial sums combined? (A7)
5. Is weight storage EEPROM or ReRAM, and which retention-drift model applies?
6. Are the batch-normalisation gain and offset programmable per chip, and what does calibration look like? (A5)
7. What are the measured magnitudes of device-to-device mismatch and cycle-to-cycle noise? (A6)
8. Is each tile's partial-sum read noise an absolute floor, independent of the signal, or proportional
   to the signal current? This decides whether multi-tile cascading has a cost. (A7, A15)
9. What are the noise, offset and gain error of the analog signal path between hidden layers
   (current mirrors, transimpedance stages)? (A15)
10. What activation frequency corresponds to a given signal-path noise figure? This is the missing link
    between the accuracy budget in report §3.4 and the throughput specification. (A18)
