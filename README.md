# NeuroGen

A continuous laminar spiking simulator and an optional trainable language/speech
system. The biological experiment uses NumPy and Matplotlib on CPU: 384 neurons,
six layers per column, two columns in each of three connected regions. The new
PyTorch system supports CPU, CUDA and ROCm, configurable sizes, joint text/speech
training and persistent streaming sessions. Its default BC-250 starting profile
has about 16 million parameters and 8,192 spiking units.

**Native C++ runtime:** the simplified spiking language core can now run through
the restored engine/column/neuron-array structure with CPU, CUDA, or Vulkan compute.
See [NATIVE.md](NATIVE.md) for builds, checkpoint-compatible usage, tests, and the
inference-only scope. Training remains PyTorch.

**For language training, speech, continuous sessions and BC-250 deployment, start
with [LANGUAGE.md](LANGUAGE.md).** GPU support must be checked on the target board;
the implementation has CPU and NVIDIA laptop validation. No pretrained language or speech
knowledge is bundled. Continuous state updates are not evidence of human thinking.
[Current validation results](LANGUAGE_VALIDATION.md) include a 96.9% alphabet-cycle
sanity check and the remaining speech/hardware limitations.

Neurons integrate excitatory/inhibitory conductances, emit spikes, adapt and
communicate through delayed synapses. Local STDP and homeostatic rules modify
weights; correlated contacts can grow and unused/weak contacts can be pruned.
This is an inspectable biological model, not a validated human brain or thinking
agent. See [the research and equations](BIOLOGICAL_DESIGN.md).

## Run

Open the desktop model workbench with `make gui`. It brings checkpoint loading,
chat, model/configuration editing, training, speech tests and run charts into one
window. See [WORKBENCH.md](WORKBENCH.md) for installation and a first test.
Fresh language training automatically prepares a SlimPajama subset; training
files are optional. See [the sampling controls](LANGUAGE.md#automatic-slimpajama-training).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
make test
make run
```

PyTorch and CUDA are not required for the biological commands below.
`make run` simulates 15 seconds and prints a unique output directory. Simulation
speed depends on hardware and chart-generation cost; simulated time is explicit.

```bash
.venv/bin/python -m neurogen simulate --duration-ms 15000 --output runs/my-run
.venv/bin/python -m neurogen simulate --resume runs/my-run/state.npz --forever --realtime
```

Open the printed `report.html` in a browser. Charts refresh every five seconds
of simulated time and at shutdown. Ctrl-C completes the current tick and saves
state and charts. Checkpoints are also saved at each reporting interval.
Each invocation uses a new, empty directory, including resume segments, so old
and new measurements cannot silently mix. The resume checkpoint supplies neuron,
synapse, clock, RNG, stimulus and observer state; duration means additional time.

## What a run produces

| File | Contents |
|---|---|
| `report.html` | Local report with auto-refreshing charts |
| `plots/activity.png` | E/I and regional rates, silent fraction, contact turnover |
| `plots/activity_<tick>.png` | Chronological activity snapshots |
| `plots/spikes.png` | Actual spike raster and 5 ms population rate |
| `plots/populations.png` | Per-cell rate histogram and layer/region heatmap |
| `plots/synapses.png` | Weight distributions and conductances over time |
| `plots/frequency.png` | Population-rate spectrum and inter-spike interval variability |
| `plots/learning.png` | Observer accuracy and cross-entropy, training/evaluation separated |
| `metrics.jsonl`, `rates.csv` | Windowed measurements and units |
| `neuron_rates.csv`, `spikes.csv` | Per-cell window rates and every spike event |
| `trials.csv` | Actual labels, predictions, losses and evaluation phase |
| `console.log`, `summary.json` | Human-readable progress and machine-readable run summary |
| `state.npz` | Atomic, pickle-free live checkpoint |

Plots retain the most recent 2,000 reports/trials and a five-second spike buffer.
CSV/JSONL logs preserve the complete run. Working buffers are bounded; log and
snapshot disk usage grows with duration. `--report-ms` defaults to 1000 and
`--plot-ms` to 5000; plot intervals must be multiples of report intervals.
Matplotlib renders headlessly, so no display server is required.

## What accuracy and loss mean

The default `--stimulus cue` drives a pseudorandom sensory column's L4 E cells
for 50 ms every 250 ms. A separate softmax observer predicts the column using
association E spikes collected from 50–150 ms. It receives no voltage, input,
clock-derived label features, or direct sensory shortcut. Labels never update
biological synapses.

Predictions and cross-entropy are measured **before** each observer update.
Every 60 trials contains 40 observer-training trials and 20 evaluation trials
that do not update the observer. The biological core continues its local,
unsupervised plasticity throughout. The observer is a diagnostic of accessible
information, not proof that Hebbian learning solves the task. With two columns,
uniform guessing gives 50% accuracy and cross-entropy ln(2) ≈ 0.693 nats.
Training results are online/prequential, not an independent test score. Evaluation
begins after ten simulated seconds; shorter runs correctly show evaluation N/A.

`--stimulus none` disables sensory pulses and the labeled task. `--stimulus
alternating` uses the earlier alternating pulse protocol without a prediction
score. Both report accuracy/loss as N/A rather than inventing values. Explicit
Poisson background afferents remain unless disabled in configuration.

## Understanding Hz

`mean_hz` is spikes **per neuron per simulated second**, averaged over the
reporting window. It is not network throughput, integration speed or a brain-wave
frequency. At 10 Hz per neuron, 384 cells emit 3,840 spikes/s in aggregate. The
1 ms timestep means 1,000 integration ticks per simulated second, independently
of how often individual cells spike.

The console reports E/I rates, the 95th percentile, network spikes/s and separate
probe metrics. JSONL additionally records wall-clock throughput. The spike raster,
rate distribution and spectrum reveal bursts and synchrony hidden by a mean.
The spectrum describes binned population spike activity, **not EEG or LFP**.
[The frequency investigation](FREQUENCY.md) explains the measurements and research.

Recalculate firing rates independently from raw spike logs:

```bash
.venv/bin/python audit_rates.py runs/my-run
```

This checks every reported mean and writes `rate_audit.json` with per-cell rate
percentiles, population spectrum, short-bin peak rate and interval variability.

## Configuration and experiments

```bash
.venv/bin/python -m neurogen simulate --config configs/biology.json --duration-ms 5000
.venv/bin/python -m neurogen simulate --stimulus none --duration-ms 5000
.venv/bin/python -m neurogen simulate --frozen --duration-ms 15000
.venv/bin/python benchmark_biology.py
```

`--frozen` disables biological weight/topology updates; the separate diagnostic
observer still follows its training/evaluation schedule. Configuration exposes
region names/projections, columns, timestep, background input, rate target,
plasticity rates and contact budgets. The cue task requires at least two regions
and two columns per region; other anatomies can use `--stimulus none`.

The benchmark separately compares plastic, frozen and disconnected cores using
an offline ridge observer; checks causal propagation and checkpoint continuation;
and runs 60-second stability experiments. It writes new results under `runs/`.
It is a different evaluation protocol from the live softmax observer. No historical
training artifacts are required to run either experiment.

## Layout

| Path | Purpose |
|---|---|
| `neurogen/anatomy.py` | Laminar populations and permitted contacts |
| `neurogen/biology.py` | Dynamics, local learning, structural plasticity, persistence |
| `neurogen/probe.py` | Independent cue observer and reproducible stimulus labels |
| `neurogen/simulation.py` | CLI runner and measurement/logging |
| `neurogen/visualization.py` | Spike buffers, charts and local HTML report |
| `neurogen/language/` | Optional trainable spiking language core, speech objectives, GPU diagnostics and streaming runtime |
| `configs/language_bc250.json`, `LANGUAGE.md` | Portable language configuration, training and deployment guide |
| `examples/make_language_demo.py` | Authored toy sequence, dialogue, arithmetic and recall curriculum |
| `configs/biology.json` | Explicit default biological settings |
| `tests/` | Dynamics, plasticity, measurement, observer and CLI tests |
| `benchmark_biology.py`, `audit_rates.py` | Controlled experiments and independent measurement audit |

The retired custom CUDA implementation is excluded from the native CMake build.
The native engine restores its C++ organization while preserving the simplified
language architecture. Historical source remnants are not part of the active build.
Old checkpoints, logs, plots, audits and obsolete training artifacts are not required. All generated
artifacts now live under ignored `runs/` directories.
