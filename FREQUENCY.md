# What the reported frequency means

**8–14 Hz is a mean firing rate per neuron, not a slow simulation clock.**
The rate calculation is correct. It was verified against every raw spike in a
fresh run, with zero discrepancy in each reporting window. The default 1 ms
timestep advances the entire network at 1,000 ticks per simulated second.

Four quantities must be distinguished:

| Quantity | Definition | Example from the current run |
|---|---|---|
| Mean neuron firing rate | Total spikes / cells / simulated seconds | 11.13 Hz over 15 seconds |
| Total network event rate | Total spikes / simulated seconds | 4,274.6 spikes/s across 384 cells |
| Instantaneous inverse ISI | 1 / time between two spikes of one cell | Up to 333.3 Hz in the final five seconds |
| Population oscillation | Frequency component of the binned population spike-rate signal | Largest component at 4 Hz in the final five seconds |

Wall-clock throughput is another quantity: plotting or a slower computer can
reduce ticks per real second without changing the equations' simulated timestep.
The console and JSONL now label these quantities separately.

## Investigation performed

Two new runs used seed 1 and the default 384-neuron anatomy and plasticity:

```bash
.venv/bin/python -m neurogen simulate --duration-ms 15000 --output runs/current-validation
.venv/bin/python -m neurogen simulate --stimulus none --duration-ms 5000 --output runs/background-comparison
.venv/bin/python audit_rates.py runs/current-validation
.venv/bin/python audit_rates.py runs/background-comparison
```

Use fresh output directories when repeating these commands. `audit_rates.py`
independently counts the recorded spike events, recomputes each window mean,
and writes `rate_audit.json`. The generated reports and logs are local, ignored
artifacts; this document records the conclusion and reproduction protocol.

The 15-second cue run produced 64,119 spikes. The whole-run median per-neuron
rate was 9.40 Hz, the 95th percentile 29.16 Hz, and the highest per-cell average
56.53 Hz. During the final five seconds, the maximum population rate in a 5 ms
bin reached **113.54 spikes/neuron/s**, despite a much smaller long-window mean.
That is bursty activity, rather than uniformly slow firing. The raster makes
coordinated bursts across regions visible.

For a matched-duration comparison, the **first five seconds** of the cue run
produced 15,428 spikes, or 8.04 Hz per neuron. Background-only simulation over
the same interval produced 10,101 spikes, or 5.26 Hz. Both used identical initial
state and Poisson RNG seed. The latter still includes explicitly configured
background afferents; it is not a disconnected, input-free brain.

The cue protocol injects a pulse every 250 ms: **4 pulses/s**. Its final-window
population spectrum peaks at 4 Hz, with harmonics. Background-only activity had
a 3 Hz peak in its short five-second sample and a lower 5 ms population peak of
44.27 Hz/neuron. These observations support an interpretation of strong
stimulus-locked responses in the cue run. They do not identify human EEG rhythms
or establish that a single spectral peak is a stable intrinsic oscillation.
The runs also differ in elapsed adaptation time when comparing their final
windows, so those final-window spectra are not a matched-age causal estimate.

The live observer ended at 50% accuracy on 20 evaluation trials, with mean
cross-entropy 0.856 nats (uniform guessing: 0.693). That measures the supervised
observer's readout of association spikes; biological synapses never received
the labels. Current evidence points to weak useful discrimination and strong
coordinated responses, rather than an insufficient integration frequency.

## What biology supports

There is no single correct firing rate for every cortical neuron. Cell type,
layer, region, behavioral state, input and measurement window matter. A
reimplementation of the Potjans–Diesmann cortical microcircuit reports excitatory
population means around **1 Hz in L2/3 and L6, 4 Hz in L4, and 7 Hz in L5**.
Those model results illustrate why a low single-neuron average can coexist with
ongoing network activity; they are not universal targets for human cortex.
[Romaro et al., original study and Table 2](https://arxiv.org/pdf/2005.03764).

Human fast-spiking interneurons can reach much higher frequencies: a slice
study measured maximal mean rates around **338 Hz** and maximal instantaneous
rates around **453 Hz**, using current injection. These are maxima in selected
cells under stimulation, not a required average for all cortical neurons.
[Wang et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5067378/).

Population oscillations also differ from individual-cell mean rates. Circuit
models can have structured population spectra while cells fire irregularly at
much lower average rates. [Bos, Diesmann & Helias](https://arxiv.org/abs/1510.00642)
analyze those relationships in a laminar microcircuit model. NeuroGen's plot is
a periodogram of population spike rate; it does not compute electrical fields.

## Decision

Do **not** increase all neuron rates just to make the console number larger.
No automatic gain, threshold or firing-rate target was raised during this work.
The default homeostatic target is already 5 Hz, while stimulated excitatory
activity often exceeds it. Raising excitation could further amplify bursts.

The useful next measurements are layer/type-specific rates, synchrony, stimulus
selectivity, and task improvement against frozen/disconnected controls. The new
plots expose those distributions and timing. Calibration should choose a
specific biological preparation and state, then fit multiple statistics together;
matching one global mean would not establish biological fidelity.

Spectrum details: latest ≤5 seconds, 5 ms bins, 200 Hz sampling, mean subtraction,
Hann window, one-sided periodogram, and a minimum one-second observation.
At five seconds the bin spacing is 0.2 Hz; short windows have coarser resolution.
ISI coefficient of variation uses cells with at least three spikes in the recent
window. Mean CV alone does not prove population asynchrony; inspect the raster
and binned rate as well. Raw CSV remains available for alternative estimators.
