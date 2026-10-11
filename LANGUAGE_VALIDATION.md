# Language implementation validation

These are laptop **CPU** results from 2026-10-10, using Python 3.14.7 and
PyTorch 2.14.1+cpu. They do not measure BC-250 performance or validate its GPU stack.

## Learning sanity check

Artifacts: [report](runs/language-current-validation/report.html),
[metrics](runs/language-current-validation/metrics.jsonl),
[configuration/provenance](runs/language-current-validation/run.json).
The reproduction command is in [LANGUAGE.md](LANGUAGE.md).

| Measurement | Before training | After 320 optimizer steps |
|---|---:|---:|
| Validation cross-entropy, nats | 5.7658 | 0.2085 |
| Validation next-symbol accuracy | 0% | 96.875% |
| Validation perplexity | 319.19 | 1.2318 |
| Excitatory spikes / unit / internal tick | 0.07541 | 0.03617 |
| Inhibitory spikes / unit / internal tick | 0.15381 | 0.19580 |

This uses the tiny 128-unit, 54,497-parameter model, a batch of 4, windows of
32 symbols, and AdamW at 0.003. Validation comprises two deterministic sampled
batches (256 next-symbol targets). The documents are different lengths of the
**same repeated alphabet cycle**; their substrings deliberately overlap. This is
a test that input-dependent learning works, not a generalization score for
natural language, arithmetic or dialogue. Uniform guessing among the 27 cycle
symbols would achieve approximately 3.7% accuracy.

After the prompt `abcdefghijklmnop`, greedy continuation produced:

```
qrstuvwxyz abcdefghijklmnopqrstuvwxyz abbcdefghijjklmn
```

This also exposes residual repetition errors. Teacher-forced accuracy is not
a guarantee of error-free autonomous generation.

An earlier prototype learned while its inhibitory population stayed silent.
Population diagnostics exposed that failure. The current model uses separate
normalized inhibitory excitability/dynamics and logs both populations in every
region. These spike fractions are **not physiological Hz** and do not establish
that the circuit matches cortical recordings.

## Default-size execution

[The default profile](runs/language-current-default-profile.json) exercised all
15,955,785 parameters / 8,192 spiking units with batch 1 and sequence length 32.
One CPU forward/backward/AdamW step completed in 4.65 seconds, with finite gradients
and approximately 768.8 MiB peak process RSS. Both E/I populations were active in
all four regions. This is a single cold-step functional measurement, not a warmed
throughput benchmark. RSS includes Python/PyTorch; it is not GPU memory usage.
The larger default training batch/sequence and paired speech require more memory.

## Speech and runtime

[Speech pipeline artifacts](runs/language-current-speech-pipeline/report.html)
exercise joint text, CTC, mel and stop losses for two optimizer steps using
synthetic tones paired with a one-byte label. They check finite computation and
pipeline integration only. The tones are **not human speech**, and the high
recognition errors are not presented as successful recognition. No speech
intelligibility or unseen-speaker quality claim is supported by this test.

Automated checks cover causal frame extraction and chunk equivalence, repeated
CTC symbols across boundaries, gradients through recognition and synthesis,
waveform reconstruction, streaming text and asynchronous WAV output, and persistent
neural/audio/sampling state. The doctor also exercises speech forward/backward.

## Verification status

- Initial CPU validation: 27 tests collected, **26 passed**, GPU test skipped because the
  installed PyTorch build is CPU-only. The language tests were rerun after the
  final runtime changes and passed (10 passed, 1 GPU skip).
- Resuming CPU training reproduces uninterrupted model parameters and sampler
  state exactly in the optimizer-resumption test.
- CPU doctor passed, including text/speech comparison and finite gradients.
- Python compilation and wheel packaging succeeded; optional language modules
  are included in the 0.5.0 package.
- Microphone/speaker hardware, real-speech learning, BC-250 ROCm operation,
  sustained GPU memory use and full-duplex acoustic latency remain untested.

For the next experiment, use genuinely disjoint natural-language/dialogue data
and paired human speech, then measure accuracy and latency on the deployment
machine. Continuous idle dynamics alone provide no evidence of human-like thought.

## CUDA laptop verification (2026-10-10)

The development environment originally had `torch 2.14.1+cpu`, despite a working
NVIDIA driver. It now uses the official `2.11.0+cu128` build (CUDA 12.8) on the
GTX 1650. [The device diagnostic](runs/language-current-cuda-diagnostic.json)
passed text and speech forward/backward checks with finite gradients. Maximum
text output difference from CPU was `3.87e-7`; maximum speech-loss difference
was `2.38e-7`. The diagnostic uses the tiny network; its selected laptop preset
specification describes sizing rather than the network used for numerical comparison.

The GUI's actual model subprocess loaded the existing alphabet-cycle checkpoint
on CUDA, generated `qrstuvwx` after `abcdefghijklmnop`, and scored text with a
finite loss. [A separate two-step training smoke test](runs/language-current-cuda-training/report.html)
used the 500,441-parameter laptop preset, batch 1 and sequence length 32. It saved
a checkpoint and plots with finite gradients and a measured peak PyTorch tensor
allocation of **38.6 MiB**. This excludes driver/context memory and is not a
sustained-memory, throughput or language-quality benchmark. The tiny synthetic
corpora deliberately share alphabet-cycle substrings.

The full suite was rerun with GPU access: **all 33 tests passed**, including the
GPU test. Dependency consistency also passed. This validates NVIDIA execution on
the laptop; BC-250 ROCm, human-speech quality and audio hardware remain untested.

## Automatic SlimPajama preparation (2026-10-10)

[The live-download smoke test](runs/language-current-slimpajama-verified/report.html)
streamed 8 training and 4 validation documents from the public
`gmongaras/SlimPajama-627B_Reupload` repository at commit
`c34c22dbb10ae6b264a2f357a909d1a537141b36`, using sampling seed 7, a four-document
shuffle buffer and a 1,024-byte text limit per document. Source details and file
hashes are saved in its [manifest](runs/language-current-slimpajama-verified/corpus/manifest.json).
The original Cerebras repository returned HTTP 401, so the default uses the
public reupload; equivalence of all original records has not been verified.

The laptop preset completed one CUDA optimizer step, saved a checkpoint and
generated charts, with finite gradients and **36.0 MiB** peak tensor allocation
(excluding driver/context memory). Dataset preparation and the process exited
cleanly. This checks the real download-to-training path; twelve documents and
one step do not establish natural-language quality.

[An offline continuation](runs/language-current-slimpajama-resumed/report.html)
loaded that checkpoint on CPU and completed step 2 with both Hugging Face offline
flags enabled. It reused the saved corpus and preserved its file hashes.

The full suite passed **41 tests**, including CUDA. The eight dataset checks were
rerun after limiting Datasets 5's concurrent shard prefetch and also passed. They
cover seed selection, real iterable shuffling with multiple source shards,
document bounds, Unicode, exact split deduplication, cancellation cleanup,
portable offline resume, tamper rejection and exact CPU optimizer continuation.
An automated hidden Tk window checked data-source selection, generated command
arguments, preparation progress, resume mode and preference restoration. Dependency
consistency passed with `torch 2.11.0+cu128` unchanged.
