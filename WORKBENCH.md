# NeuroGen desktop workbench

Native C++ inference now supports `native-cpu`, `native-cuda`, and `vulkan`.
See [NATIVE.md](NATIVE.md) for build instructions, unchanged checkpoint/session
formats, and backend limitations. Training continues to use PyTorch devices.

Run from this project directory in a graphical desktop session:

```bash
make gui
```

Or preload the included alphabet-cycle checkpoint:

```bash
.venv/bin/python -m neurogen gui --checkpoint runs/language-current-validation/checkpoint.pt
```

The GUI uses Python's Tkinter/Tk. If importing Tkinter fails because the runtime
libraries are missing, install the operating system's Tk package. On this Arch
Linux laptop, the required command is:

```bash
sudo pacman -S --needed tk
```

On Debian/Ubuntu, the usual package is `python3-tk`. Use a Python installation
whose Tkinter extension matches the installed Tk libraries. Language operations
also require the appropriate PyTorch build described in [LANGUAGE.md](LANGUAGE.md).

## First test

1. Choose `runs/language-current-validation/checkpoint.pt` and Auto, then
   click **Load model**. Passing `--checkpoint` above loads it automatically.
2. Open **Chat**, select **Text continuation**, and enter `abcdefghijklmnop`.
3. Set temperature to `0` for greedy decoding, then click **Send** or Ctrl+Enter.
4. Change temperature, top-k or the output limit to compare continuations.

That included checkpoint learned a repeated alphabet sequence. Use a separately
trained dialogue checkpoint for conversational tests. **Score text** measures
fresh-state next-byte cross-entropy, perplexity and accuracy without changing the
active conversational state.

For NVIDIA acceleration, install the CUDA PyTorch build following
[LANGUAGE.md](LANGUAGE.md#install-and-check-the-deployment-machine), restart the
GUI, select `cuda`, and click **Load model** again. Changing the device picker
does not move a model that is already loaded. Check the model summary for the GPU
name; **Runs & diagnostics → hardware check** exercises text and speech gradients.
On the 4 GiB GTX 1650, start new training with the `laptop` preset, batch 1,
sequence length 32 and a 3072 MiB GPU budget.

## Controls

- **Chat:** streamed responses, cancellation, sampling settings, optional
  continuous idle activity, manual idle steps, session save/restore and JSON chat
  export. A session preserves model state and its conversation sidecar and must
  be restored against the same checkpoint.
- **Model & configuration:** choose a size preset, edit and validate JSON, inspect
  parameter/memory estimates, create a random model, and export its weights.
  Changing architecture creates a fresh model; a loaded checkpoint keeps the
  dimensions its weights were trained with.
- **Train:** automatically create a fresh SlimPajama training/validation subset,
  with configurable repository, revision, document counts, size limit and seed.
  Leave the data seed blank for a new sample each fresh run. Choose **Local files**
  for custom text/dialogue data and add optional speech manifests. Choose fresh training, initialization from checkpoint weights, or
  full optimizer resume. Configure steps, batch/sequence sizes, accumulation,
  learning rate, evaluation, speech losses and memory budget. Full resume needs
  matching original data/settings; SlimPajama resume reuses the checkpoint's
  adjacent `corpus/` folder offline. Weight exports use initialization mode.
- **Speech:** transcribe a WAV, record the microphone, move a transcript into
  Chat, synthesize text to WAV, and play the output. Recognition/synthesis need
  paired-speech training. Recording/playback require the optional `voice`
  dependencies and working audio devices. Recording stops at 30 seconds.
- **Biological simulation:** edit biological JSON, load biological `.npz`
  checkpoints, control stimulus/plasticity/timing, and start or stop a simulation.
- **Runs & diagnostics:** hardware checks, live logs, plots and latest metrics,
  HTML reports, checkpoint discovery, worker restart and job stop controls.

Model operations run in an isolated process; training and simulation use managed
subprocesses. Before an experiment unloads a saved chat model, its session is
saved automatically under `runs/workbench/sessions/`. Training stops at a completed
optimizer step and biological simulation at a completed tick. Stopping during
dataset preparation removes partial corpus files; no model has been trained yet.
Force stop can lose
work since the last checkpoint. Closing waits for active processes to stop.

Settings, experiment arguments and logs live under `runs/workbench/`. Newly
created models have random weights until trained. Model exports contain weights;
full training checkpoints also contain optimizer and RNG state.

## Validation

The workbench backend tests exercise create/save/load, streamed generation,
scoring, cancellation, recovery after a failed load, session replay and starting
a training job from an exported model. A hidden Tk window also passed automatic
and local corpus selection, training argument validation, download progress,
resume selection and preference restoration checks. Interactive layout and audio
devices still need local validation.
The model subprocess also passed a CUDA checkpoint load, streamed continuation
and scoring check on the GTX 1650 with `torch 2.11.0+cu128`; see
[LANGUAGE_VALIDATION.md](LANGUAGE_VALIDATION.md#cuda-laptop-verification-2026-10-10).
