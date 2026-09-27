# Machine-Sound Anomaly Detection on a 30 × 30, 6-bit Analog Chip — Short Report

*Version 2, 23 September 2026 · KaiwenLin — KaiwenLin@utexas.edu*
*Details, tables and all four figures: [full report](docs/REPORT_full.md) (Chinese:
[REPORT_full-zh.md](docs/REPORT_full-zh.md)). Every assumption: [assumptions.md](assumptions.md).*

> **In one paragraph.** Can a small analog AI chip — a 30 × 30 grid of 6-bit weights, with no digital
> conversion between layers — detect faulty machines by their sound? In simulation, yes: a standard detector
> that needs 380 such grids can be shrunk to 6 and keep about 80 % of its detection ability. It only works if
> the model is trained with the chip's imperfections simulated. Removing the converters between layers — the
> chip's main design bet — costs almost nothing; the noise of the analog signal path is what sets the limit.
> And the converters, not the weights, decide how many bits the chip needs. Nothing here has been tested on
> real silicon.

---

## The question

The chip computes one neural-network layer per 30 × 30 grid of stored weights — a *tile* — in analog
current. Weights sit inside the memory cells, inputs arrive as currents, and the multiply-and-add happens in
the circuit itself rather than in a processor. Between layers the signal stays analog: the published design
avoids converting it to digital and back, which saves power and area but lets errors pass from layer to layer
uncorrected.

I asked three things:

1. **Does a useful anomaly detector fit?** The public benchmark model needs 380 tiles.
2. **What does each hardware limit cost?** Only 30 input lines, 6-bit precision, imprecise weight
   programming, chip-to-chip variation, analog noise, and no converters between layers.
3. **What must the hardware guarantee, and what must the training guarantee,** so that *every* chip works —
   not just the average one?

## How I studied it

**The task.** I started from the anomaly detector of MLPerf Tiny, the standard benchmark for small
machine-learning devices, on toy-car recordings from the DCASE 2020 challenge. The detector is an
autoencoder: it is trained only on healthy machines, learns to compress and rebuild their sound, and flags a
recording it cannot rebuild well. Before changing anything I reproduced its published result.

**Shrinking it.** The benchmark model has 10 layers; its first layer alone needs 110 tiles, because it reads
640 input values. How a layer is split across tiles matters. Splitting the *outputs* across tiles is cheap —
the tiles work independently. Splitting the *inputs* is not: several tiles must add their partial results on
one wire, and each adds its own errors. So I cut the network to 6 layers of at most 30 × 30, one tile each,
with no splitting at all. The price is input: 30 values instead of 640 — for example 6 frequency bands over
5 moments in time.

**Simulating the chip.** A software model of the chip adds, to every layer:

- rounding of weights, signals and biases to 6 bits;
- weight-programming error — every stored weight is a little off, and differently on every read;
- chip-to-chip variation — a fixed error pattern that each chip carries for life;
- noise on the analog signal between layers;
- converters (D/A in, A/D out) only where I place them: at the two ends of the network, as in the chip, or
  around every layer, as in conventional designs.

**Chip-aware training.** The usual approach trains a clean model and loads it onto the chip. The alternative,
*hardware-aware training*, puts the simulated imperfections into training, so the model learns to tolerate
them. I compared both.

**Measuring.** Detection quality is AUC: 0.5 is a coin flip, 1.0 is perfect. Each result is averaged over
10 simulated chips and 3 independent trainings, and I always look at the worst chip as well as the average,
because a customer receives one chip, not the average of ten. When comparing two designs I compare them on
the same training runs, so that luck in training does not pass for a difference between designs.

**Checking the simulator.** Finally I re-ran the key results with a more realistic chip model — converters
with fixed ranges, an activation that saturates like real circuits, and three different kinds of analog
noise — to see which conclusions depend on my modelling choices.

## The trade-offs at a glance

| Trade-off | Answer | What it means for the chip |
|---|---|---|
| Model size vs accuracy | 380 → 6 tiles keeps ~80 % of the detection ability (AUC 0.785 → 0.72) | One small chip can host the task; no layer needs more than one tile |
| 30 input lines: frequency detail or time context? | **Time.** 6 bands × 5 moments beat 30 bands × 1 moment by +0.07 AUC | Give a few bands a short memory; simple leaky integrators (τ ≤ ~128 ms) work |
| Weight-programming precision vs yield | With chip-aware training, 3 % programming error costs nothing; without it, some chips do worse than a coin flip | ~3 % program-verify tolerance is enough — *if* the model is trained for the chip |
| Converters between layers or not | Removing them costs 0 to 0.01 AUC; with realistic fixed-range converters, nothing measurable | The no-converter design holds; the limit is analog noise — 40 dB of dynamic range was still enough |
| Bits | Weights and biases work at **4 bits**. The two converters need 5 — or 4, if each input's average level is subtracted before conversion | Cheaper weight programming (16 levels instead of 64); converter bits depend on one small front-end feature |
| Per-chip calibration vs training recipe | Adding noise during training makes every chip robust; per-chip calibration adds almost nothing | Put the training recipe in the spec; skip per-chip calibration; set the alarm threshold on a real chip |

## The findings, one by one

### 1. It fits, at a moderate cost

The 6-tile model scores AUC 0.72 on the simulated chip, against 0.785 for the full benchmark model and 0.72
for the same small model in ideal digital arithmetic. Measured as distance above a coin flip, it keeps about
80 % of the full model's ability, with 63 times fewer weights. The simulated chip itself costs nothing
measurable on top of the shrinking: the loss comes from the smaller model, not from the analog hardware.

### 2. Spend the input lines on time, not on frequency detail

With only 30 inputs, one option is 30 finely spaced frequency bands at a single moment; another is 6 coarse
bands, each seen at 5 successive moments. The second wins clearly, by +0.07 AUC, and every option with at least
two moments beats both single-moment options. Machine faults show up as rhythm and change — a knock, a
wobble — more than as a static spectrum.

Storing past moments needs memory. A cheaper analog alternative is a *leaky integrator*: a circuit that
remembers a smoothly fading average of its input, with a time constant τ. Replacing the 5 stored moments by 5
integrators per band with τ between 32 and 128 ms works as well as storing frames (slightly better in ideal
arithmetic, within noise on the simulated chip). Longer time constants, up to half a second, make it worse.
This matters for the front end: short time constants are the cheap ones.

### 3. Chip-aware training decides yield

Weight programming is never exact. With chip-aware training, AUC holds at 0.71–0.74 for programming errors
up to 3 %, the same as the ideal model. Without it, the *average* chip still looks acceptable — 0.62 to 0.69 —
but the *worst* of ten chips falls below a coin flip from 1 % programming error onward (Figure 2). Chip-aware
training narrows the spread between chips up to 6×.

The practical reading: a program-verify tolerance of about 3 % is enough, and tightening it buys no accuracy
while costing test time — but only if the model is trained for the chip. Trained the usual way, the same
hardware would ship some chips that do not work.

![Figure 2](constraint-study/results/figures/fig2_noise_stefix.png)

*Right panel: the worst of ten simulated chips. Trained normally (red), it falls below a coin flip once
weight-programming error reaches 1 %. Trained with the chip's imperfections simulated (blue), it holds.*

### 4. Removing the converters between layers is nearly free

This is the chip's central design choice, so I compared it head-on with a conventional design that converts
at every layer boundary — 12 conversions instead of 2. With signal noise of up to about half a converter step,
dropping the inner converters costs about 0.01 AUC. With converters modelled realistically — a fixed range
set once, instead of re-ranging for every batch of data as my first simulator did — the cost disappears into
the noise.

What does matter is the noise of the analog signal path. When it grows to about one converter step, *both*
designs fail together. So the question for the hardware is not where to place converters but how quiet the
signal path is. How much noise is tolerable depends on its nature. If noise grows with the signal, the budget
is about half a converter step. If it is a fixed floor, the network learns to use more of the available
signal range to rise above it, and a dynamic range of 40 dB was still enough; the true limit is lower and I
have not found it yet. If noise scales with each individual signal value, removing the inner converters does
cost about 0.016 AUC. Which of these the real chip follows is the first question I would ask.

### 5. The converters, not the weights, set the bit width

Varying one precision at a time shows where the bits are needed (Figure 4). Weights work at 4 bits (−0.01
AUC) and so do biases — 16 programmable levels per cell instead of 64, which makes programming easier. More
than 6 bits buys nothing anywhere.

The two converters are the real floor: at 5 bits they cost nothing, at 4 bits detection falls to 0.57. The
reason is simple. Sound features sit at a large constant level with small fluctuations on top — roughly
−26 dB with a few dB of movement. A converter must cover the constant too, so at 4 bits its step is larger
than the fluctuation it should carry. Subtracting each input's average level before the D/A converter, and
adding it back after the output converter, shrinks the needed range 3.6 times. With that offset, 4-bit
converters cost nothing, and a chip with 4-bit weights, biases *and* converters matches the 6-bit one
(0.723 against 0.722). Whether that offset is cheap in the front end decides whether the converters can
drop to 4 bits.

![Figure 4](constraint-study/results/figures/fig4_bits_v2.png)

*Left: all three precisions lowered together; the grey dashed line is version 1, before the bug fix. Right:
one precision lowered at a time — weights (blue) and biases (purple) are fine at 4 bits; the converters
(orange) need 5 bits, or 4 with the offset subtracted (dashed).*

### 6. Train with noise, skip per-chip calibration, measure the threshold on silicon

Adding a little noise to the signals during training makes the no-converter model robust on every chip —
even on a perfectly quiet one, where the model trained without noise loses 0.09 AUC. It is not about matching
the chip: the same trick helps a model in ideal digital arithmetic too. It is a regulariser, and it belongs in
the specification alongside the weights.

Calibrating each chip after manufacturing — re-tuning with a minute or more of recorded normal sound — then
adds almost nothing (+0.001 to +0.006 AUC), and it cannot rescue a model trained the wrong way. That step and
its test time can leave the production flow.

A deployed detector also needs an alarm threshold. Set in simulation, it fails on real chips: chip noise
raises every score by about 1.8×, so a simulated threshold flags everything. Set on one reference chip, the
same threshold works on the other nine, holding the false-alarm rate near its 10 % target. With the input
offset from finding 5, the shift almost disappears.

## Does the simulator itself matter?

Every conclusion above rests on a model of the chip, so I changed the model and re-ran the key results:

- **Converter ranges fixed once, as in real hardware**, instead of re-ranged for every batch: the
  no-converter design is unchanged; the conventional design loses a little, because part of its advantage came
  from the unrealistic re-ranging.
- **A saturating activation**, like a current mirror limited by its bias, or like a differential pair,
  instead of the ideal ReLU: no measurable difference.
- **Three kinds of analog noise** — a fixed floor, noise proportional to each signal, and shot noise that
  grows with the square root of the current: the conclusions hold under all three, except that noise
  proportional to each signal costs about 0.02 AUC and makes removing the inner converters cost 0.016.

So the conclusions do not hinge on the modelling choices I was least sure of — with one exception, the kind of
analog noise, which only a measurement on the chip can settle.

## What the ML side hands to the hardware side

For the 6-tile model the whole hand-off fits on a page: the tile map (one layer per tile); 4,242 weight and
bias codes; one scale per tile; a gain and offset per channel for the normalisation step; the fixed ranges
of the two converters, plus the per-input offsets if the converters are to drop to 4 bits; the alarm
threshold, measured on a reference chip; and the noise levels the model was trained against. That last item
is easy to leave out of a weights file and expensive to rediscover: a model is only valid on chips no noisier
than it was trained for.

## What I got wrong, and fixed

Version 1 of this report (12 September) said 4-bit precision fails completely. That was a bug in my own
training code: values at the top of the 4-bit range received no learning signal, so those models never
trained. I found it while investigating why the 4-bit result looked so strange, fixed it, added tests that
catch it, and retrained every affected result (114 training runs). Only that finding was wrong. One other —
the cost of removing inner converters — needed its numbers restated, from "at most 0.01" to "about 0.01".
The rest stood.

## What this does not show

- **It is simulation only.** Noise levels are assumptions or stress tests, not measurements of the real chip.
- **One benchmark, and a modest one.** At the 10 % false-alarm rate the detector catches about 37 % of faults;
  if one recording in ten is faulty, about 30 % of alarms are real. Those absolute numbers belong to toy-car
  sounds at AUC 0.72, not to the chip — the *relative* costs above are what should transfer. Industrial
  recordings (MIMII) would be a stronger test.
- **Not modelled:** power, temperature, long-term drift of stored weights, gain or offset errors in the analog
  path, and a front end simulated from the raw waveform.
- **Statistics:** three trainings per condition; differences below about 0.02 are not claimed unless they
  are consistent across trainings.

## What I would ask the hardware team

1. **Is the analog read noise a fixed floor, or proportional to the signal?** It decides the noise budget and
   whether removing inter-layer converters stays free.
2. **Is subtracting a per-input offset before the D/A cheap in your front end?** It decides whether the
   converters can drop from 6 to 4 bits.
3. **How many 30 × 30 arrays are on a die, and can one drive the next directly in analog?** The 6-tile design
   depends on it.
