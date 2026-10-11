# Trainable language and speech

NeuroGen now has an optional PyTorch language system alongside the conductance
simulator. It trains its own weights; it does not contain a pretrained LLM, ASR
service, or text-to-speech service. The executable training and streaming paths
are implemented. General conversation, intelligible speech, long-term memory,
and human-like reasoning have **not** been demonstrated.
See [LANGUAGE_VALIDATION.md](LANGUAGE_VALIDATION.md) for measured results and test scope.

## Two models with different purposes

`simulate` remains the physical-time NumPy experiment with conductance neurons,
delayed synapses, STDP, homeostasis and structural plasticity. Its equations,
biological evidence and limitations are in [BIOLOGICAL_DESIGN.md](BIOLOGICAL_DESIGN.md).

`language` is a **hybrid supervised research model**. Each region contains the
same six-layer, 64-cell column layout (50 excitatory, 14 inhibitory). Local
connections have fixed presynaptic signs and a laminar mask, with excitatory
horizontal links between adjacent columns. Input enters L4 excitatory cells;
local recurrent memory feeds deep layers/L1. Threshold events, a one-tick
refractory gate, adaptation and decaying spike traces carry recurrent state.
Inhibitory cells have faster normalized membrane dynamics and a lower threshold.
These are engineering parameters, not a fit to human recordings.

Spikes use surrogate gradients. Both graded membrane voltage and spike traces
feed the readout, which makes this explicitly a hybrid model. Dense regional
projections, LayerNorm, slow learned recurrent memory and the GRU acoustic
decoder are conventional machine-learning components. Regional projections are
not constrained by Dale's law. The language model uses AdamW/backpropagation;
it does **not** silently substitute that for biological STDP or claim structural
rewiring. The biological simulator remains the place to test those mechanisms.

Regions communicate during idle updates, with external sensory input removed.
The acoustic stream and conversational stream share weights but have separate
recurrent states; decoded transcripts enter the conversation. This is an
explicit transcript bridge, not a demonstrated unified biological speech circuit.

## Install and check the deployment machine

The base simulator still needs only NumPy and Matplotlib. Install a PyTorch
build appropriate to the machine **before** installing the language extra.
For a CPU development environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[language]'
.venv/bin/python -m neurogen language doctor --device cpu
```

For an NVIDIA laptop, install a CUDA-enabled build instead. For the GTX 1650
development laptop, use the official CUDA 12.8 wheel index:

```bash
mkdir -p runs/.pip-tmp
TMPDIR="$PWD/runs/.pip-tmp" .venv/bin/python -m pip install --upgrade 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[language]'
.venv/bin/python -m neurogen language doctor --device cuda --preset laptop
```

The explicit version replaces the previously installed `2.14.1+cpu`; an
unversioned `--upgrade` can leave that higher-version CPU build installed.
The disk-backed temporary directory avoids the development laptop's small
RAM-backed `/tmp` quota; CUDA installation stages several gigabytes of packages.
Other machines should select a compatible build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).
Installing a system CUDA toolkit does not add CUDA support to a CPU-only torch
wheel. The doctor reports the Python executable, torch path and compiled CUDA/HIP
version so an environment mismatch is visible. A GPU build with an inaccessible
driver reports the initialization error separately.

After replacing torch, restart the GUI, select `cuda`, and reload the checkpoint.
Device selection applies when loading/creating a model or starting a job; it does
not move an already loaded model. The model summary shows the actual GPU in use.
For new training on this 4 GiB laptop, start with the `laptop` preset, batch 1,
sequence length 32 and a 3072 MiB memory budget. This is a starting configuration,
not a guarantee against OOM. A loaded checkpoint retains its original size.

On the BC-250, install a working platform-specific GPU stack, then run:

```bash
.venv/bin/python -m neurogen language doctor --device rocm
```

BC-250 GPU support is a material deployment prerequisite. The community
[bc250-rocm implementation](https://github.com/akandr/bc250-rocm/blob/main/README.md)
reports a patched driver/firmware/ROCm stack and a PyTorch build targeting
`gfx1013`; stock PyTorch/ROCm packages are not sufficient for that setup.
An unlocked core count alone does not establish compatibility. This is a
community implementation, not an AMD-supported qualification of this project.
NeuroGen does not change firmware, drivers or `HSA_OVERRIDE_GFX_VERSION`.

ROCm intentionally uses PyTorch's `torch.cuda` interface, as documented in
[PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).
`--device rocm` requires a working HIP build; `--device cuda` requires CUDA.
Explicit GPU requests fail if unavailable. `auto` selects a detected GPU or
reports CPU in the run metadata. The doctor executes text and speech
forward/backward operations, checks finite gradients, and compares with CPU.
Run it on the BC-250 itself: passing on the laptop cannot validate that board.

## Size and memory

| Preset | Columns per region | Regions | Width | Spiking units |
|---|---:|---:|---:|---:|
| tiny | 1 | 2 | 32 | 128 |
| laptop | 4 | 3 | 128 | 768 |
| bc250 (default) | 32 | 4 | 768 | 8,192 |
| large | 64 | 4 | 1,024 | 16,384 |

The default has **15,955,785 trainable parameters**. FP32 parameters, gradients
and Adam's two moment arrays require approximately 243.5 MiB together.
That is not total training memory: recurrent activation graphs, speech lengths,
workspaces, allocator reservations and the operating system also consume memory.
Training defaults to FP32, batch 2, length 96 and accumulation 4. Text effective
batch size is 8; paired speech uses one variable-length example per optimizer
step. The default GPU allocation budget is 6 GiB, leaving substantial nominal
headroom on a shared-memory board, but actual capacity depends on its setup.

The preflight estimate covers text activations only. Paired speech can cost more;
start with short clips. GPU peak allocations are recorded, and exceeding the
budget stops after a completed step and checkpoint. This is an allocation guard,
not a hard allocator quota or a guarantee against OOM. Reduce batch size, sequence
length, clip duration or the model preset when necessary. Increase accumulation
to recover text effective batch size. There is no mixed-precision requirement.

`--config configs/language_bc250.json` makes the architecture explicit. A JSON
file can override `columns`, `regions`, `width`, `ticks`, `idle_steps`, `mel_bins`
and `speech_width`. Resume restores the checkpoint architecture; changing its
size requires a new training run. More columns alone do not imply better learning.

## Automatic SlimPajama training

Fresh training automatically streams a SlimPajama subset and creates separate
`corpus/train.jsonl` and `corpus/validation.jsonl` files inside its new run
directory. No training file is required. The `language` extra includes Hugging
Face Datasets; update an existing installation with:

```bash
.venv/bin/python -m pip install -e '.[language]'
.venv/bin/python -m neurogen language train --preset laptop --device cuda \
  --batch-size 1 --sequence-length 32 --memory-budget-mib 3072
```

In the GUI, open **Train**, leave **Text data** set to **SlimPajama (automatic)**,
adjust the document counts and training settings, and click **Start training**.
Logs show dataset preparation before optimizer steps begin.

Defaults are 2,000 training documents, 128 validation documents, and at most
8,192 UTF-8 bytes of text per document. Change these with `--train-documents`,
`--validation-documents` and `--max-document-bytes`, or the GUI controls.
Streaming avoids downloading the entire corpus, though remote Parquet reads
can transfer substantially more than the saved subset. `--shuffle-buffer`
defaults to 128 truncated documents. Sampling randomizes shard order and uses
this bounded document shuffle buffer. One shard reader is active at a time;
initial network reads can take time.

The default source is the public
[SlimPajama reupload](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload).
The original Cerebras repository returned HTTP 401 during integration testing;
the source is configurable with `--dataset-id` / **Dataset repository**. This
uses the uploader's SlimPajama reupload, not a separately verified copy of every
original record. Review the source's provenance and terms for your intended use.
Private or restricted repositories require access through `hf auth login` or
`HF_TOKEN`. Downloads require internet access and use Hugging Face's cache.

Every fresh run chooses a new random sampling seed. Set `--dataset-seed` (or
**Data seed**) for repeatable sampling; also set `--dataset-revision` to the
recorded commit to reproduce a previous source version. The subset uses the
repository's separate train/validation splits and excludes identical truncated
documents across and within splits. This does not detect near-duplicates.
`corpus/manifest.json`, `run.json` and checkpoints record the source commit,
sampling seed, counts and file hashes.

Optimizer resume reuses the saved subset without downloading new data. Keep the
`corpus/` folder beside `checkpoint.pt`, including when copying to the BC-250.
To train existing weights on a fresh subset instead, use `--initialize-from`.
SlimPajama supplies plain text, not paired speech or a dialogue curriculum.

## Local text, dialogue and delayed recall

Text files contain one document per nonempty line. JSONL supports either:

```json
{"text":"A document to learn by next-byte prediction."}
{"messages":[{"role":"user","content":"What is 2 + 3?"},{"role":"assistant","content":"5"}]}
```

UTF-8 bytes use IDs 0–255; explicit padding, beginning/end and role markers use
256–263. This supports arbitrary Unicode without fitting a vocabulary. A token
in reported accuracy/throughput is a byte or control symbol, not an LLM subword.
Dialogue loss supervises assistant content and its end marker; user/system
messages supply context. Plain text supervises every next symbol.

For custom data, provide **separate training and validation files** using
`--train` and `--validation`, or choose **Local files** in the GUI. These options
select local data automatically unless `--data-source slimpajama` is explicit.
Exact duplicate documents
across splits are rejected. That does not detect paraphrases or near-duplicates:
make subject/source/speaker splits deliberately. Token caches are memory-mapped;
document indices and duplicate hashes remain in RAM. Input records are parsed
one at a time. Windows never cross documents. Each sampled window starts with
fresh neural state; gradients span that window, not the entire corpus. Choose
sequence lengths long enough to include the context required by your tasks.

Generate the included small curriculum (arithmetic, dialogue and delayed recall)
and the easier next-symbol sanity check:

```bash
.venv/bin/python examples/make_language_demo.py
.venv/bin/python -m neurogen language train --preset tiny --device cpu \
  --train runs/demo-data/sequence-train.jsonl \
  --validation runs/demo-data/sequence-validation.jsonl \
  --steps 320 --batch-size 4 --sequence-length 32 --accumulate 1 \
  --learning-rate 0.003 --eval-every 40 --eval-batches 2
```

For the authored dialogue curriculum, use `train.jsonl` and `validation.jsonl`
from that directory, and a longer sequence (e.g. 128). This is a harder experiment;
the existence of the examples does not establish successful arithmetic or recall.
For substantial language ability, provide a much larger, varied, appropriately
licensed text/dialogue corpus with genuinely held-out evaluations.

A conservative initial BC-250 training command, after the doctor passes:

```bash
.venv/bin/python -m neurogen language train --device rocm --preset bc250 \
  --batch-size 1 --sequence-length 64 --accumulate 8 --steps 10000
```

## Paired speech training

Use mono 16-bit PCM WAV files at 16 kHz. Convert other formats explicitly:

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 -c:a pcm_s16le converted.wav
```

Speech JSONL manifests contain paths relative to the manifest and transcripts:

```json
{"wav":"clips/example.wav","text":"The words actually spoken in this recording."}
```

Add `--speech-train speech/train.jsonl --speech-validation speech/validation.jsonl`
to the text training command. Both are required together. Split speakers or
recording sessions, not adjacent pieces of the same recording. Identical decoded
waveforms across the two manifests are rejected. Default maximum clip duration
is six seconds; unsupported sample formats and impossible CTC alignments fail
explicitly. CTC requires enough frames for every transcript byte, including an
extra separation frame for consecutive repeated bytes.

The joint objective is:

```
text cross-entropy + 0.1 * CTC + 1.0 * mel L1 + 0.1 * stop BCE
```

Weights are configurable. Recognition uses causal 512-sample windows, a 320-sample
hop (20 ms), 64 log-magnitude mel bands, and the shared recurrent core. Its CTC
alphabet contains bytes plus blank. Synthesis encodes text through the core and
trains an attentive GRU to predict acoustic frames and an end signal. A small
inverse-mel/Griffin-Lim vocoder reconstructs PCM. It is a baseline with known
quality limitations, not a modern pretrained neural vocoder. The speech-trained
flag means positive paired objectives have received updates, not that speech is
accurate or intelligible. Speech commands reject text-only checkpoints.

Validation logs CTC loss, mel error, stop loss, greedy character error rate and
word error rate. CER/WER are case-sensitive and can exceed 100% with insertions.
Mel validation uses teacher forcing; it does not establish free-running speech
quality. Evaluate generated recordings and real speech transcripts separately.

## Continuous sessions and real-time I/O

```bash
.venv/bin/python -m neurogen language chat --checkpoint runs/YOUR-RUN/checkpoint.pt
.venv/bin/python -m neurogen language stream --checkpoint runs/YOUR-RUN/checkpoint.pt \
  --save-session runs/my-session.pt --voice-output runs/voice
.venv/bin/python -m neurogen language listen --checkpoint runs/YOUR-RUN/checkpoint.pt --wav input.wav
.venv/bin/python -m neurogen language speak --checkpoint runs/YOUR-RUN/checkpoint.pt \
  --text 'Hello.' --output runs/hello.wav
```

Use `--voice-output` only with a speech-trained checkpoint. It synthesizes completed
responses in a separate worker while the main session can receive new input and
advance its state. New input interrupts byte generation, cancels pending voice
work, and emits `audio_cancel` so a playback client can stop its audio. Output
currently buffers a complete utterance into a WAV; it is not an incremental
low-latency vocoder. Microphone capture and speaker playback belong to the client.

An optional local client is included:

```bash
.venv/bin/python -m pip install -e '.[voice]'
.venv/bin/python examples/voice_client.py --checkpoint runs/YOUR-RUN/checkpoint.pt
```

It uses SoundDevice/PortAudio for microphone capture and playback, a configurable
RMS gate, 0.6-second pause detection, and a five-second maximum input turn. Use
headphones: acoustic echo cancellation is not implemented. Audio overflow stops
the client instead of silently dropping samples. Device I/O has not been tested
on the laptop or BC-250; the underlying PCM/JSON/WAV pipeline has automated tests.

`chat` accepts one text turn per line and `/reset`, `/save`, `/quit`. `stream`
accepts newline-delimited JSON, with flushed newline-delimited JSON responses:

```json
{"type":"text","text":"Hello"}
{"type":"audio","pcm16_base64":"...","final":false}
{"type":"audio","pcm16_base64":"","final":true,"respond":true}
{"type":"tick","steps":3}
{"type":"save"}
{"type":"reset"}
{"type":"quit"}
```

Audio chunks are mono PCM16 little-endian at 16 kHz, at most one second per event.
An utterance is limited to 30 seconds before its final boundary.
The client marks the utterance boundary; no VAD or acoustic echo cancellation is
included. Transcript deltas preserve CTC collapse and UTF-8 state across chunks.
Events include `ready`, `transcript_delta`, `transcript`, `text_delta`, `turn_end`,
`audio_output` (WAV path and timing), `audio_cancel`, `activity`, and `error`.
`--max-new-tokens` bounds generated bytes; input/voice queues are bounded.
On EOF, current generation/voice work finishes. A client must handle backpressure.

Idle updates run between inputs (default requested cadence 10 updates/s), and
configured `idle_steps` also run between tokens during training. This supplies
continuous recurrent dynamics and training pressure across short silent gaps.
It does not establish private reasoning, planning, consciousness or indefinite
memory retention. Long untrained idle intervals can change or erase context.
Timing is best-effort: inference/vocoder real-time factors and idle update latency
must be measured on the target. A factor below 1 means faster than audio duration.

Session files preserve neural state, spike counters, sampling RNG, partial UTF-8,
CTC state and buffered PCM features. They require the exact model checkpoint hash.
`--resume-session` restores those states; pending jobs, playback and the application's
partially assembled transcript are not resumed. Save at a completed turn boundary
for conversational continuity. `/reset` explicitly starts a fresh session.

## Checkpoints and measurements

Each run uses a new output directory. `checkpoint.pt` contains the model,
optimizer, sampler/RNG, architecture, training settings, dataset hashes and step.
Pass `--resume checkpoint.pt`, the same data/settings, and a new `--output`;
`--steps` is the desired total step count. Resumption is tested for exact CPU
parameter equality. Cross-device numerical equality is not promised. Ctrl-C
finishes the optimizer step and saves. GPU driver failure/OOM may require using
the previous complete checkpoint. Checkpoints load with `weights_only=True`.

A run produces `metrics.jsonl`, `run.json`, `learning.png`, `report.html`, token
caches and `checkpoint.pt`. Metrics include validation loss/accuracy/perplexity,
E/I and per-region spike fractions, gradient norm, supervised-token throughput,
GPU peak allocation and optional speech losses/error rates. Source and data
hashes and the actual backend are recorded. Charts update on evaluation intervals;
raw history remains available. Checkpoints store metric history, so very long
runs should use a larger evaluation interval to limit history size.

**Language spike activity is spikes per unit per internal tick, not Hz.**
These normalized token/audio updates have no calibrated biological millisecond.
`--idle-hz` is a scheduler setting. Physical neuronal Hz remains specific to
`simulate`. Do not compare the two numerical values directly.

## Research grounding

- The laminar scaffold draws on the sources and caveats in
  [BIOLOGICAL_DESIGN.md](BIOLOGICAL_DESIGN.md); a 64-cell column is an abstraction,
  not a reconstruction of a human cortical column.
- [Neftci, Mostafa & Zenke (2019)](https://arxiv.org/abs/1901.09948) motivates
  surrogate-gradient credit assignment in spiking networks. Our fast-sigmoid
  derivative and hybrid voltage readout are implementation choices, not evidence
  that cortical neurons perform backpropagation.
- [Graves et al. (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) provides
  CTC for learning sequence labels without frame-level alignments. The
  [PyTorch CTC contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
  underlies the alignment-length checks and blank handling.
- [Wang et al., Tacotron (2017)](https://arxiv.org/abs/1703.10135) motivates paired
  text/acoustic prediction. This small GRU decoder and deterministic vocoder are
  not a reproduction of Tacotron or its reported voice quality.

The next evidence gates are held-out natural-language perplexity, multi-turn
recall across measured idle delays, speech CER/WER on unseen speakers, intelligible
free-running synthesis, and sustained latency/memory measurements on the BC-250.
Passing the included toy task establishes a learning path, not those capabilities.
