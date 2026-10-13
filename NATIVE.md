# Native C++ LIF execution

The simplified language model now has a C++17 inference engine. It restores the
historical `NeuralEngine` → `CorticalColumnV2` → `ALIFNeuronArrays` organization,
without restoring the retired compartmental model, custom token trainer, or old
CUDA-only dependencies. The public headers compile without CUDA or PyTorch.

The reference is `neurogen/language/model.py` at `fee31aa`. Both implementations
use the same 64-cell, six-layer column, E/I populations, masked signed softplus
recurrent weights, L4 input, deep-layer feedback, adjacent-column excitation,
one-tick refractoriness, adaptation, graded voltage/trace readout, layer norm,
slow memory, and sequential regions. Idle updates remove only external input to
the first region. Vocabulary, checkpoints, and six-field session state are unchanged.

## Scope

| Component | Execution |
|---|---|
| Text embedding, column dynamics, projections, regional memory, token logits | C++ native runtime |
| Dense projections and synchronous recurrent ticks | CPU, optional CUDA kernels, or optional Vulkan compute shaders |
| Activation functions, layer norm, memory update, region orchestration | C++ CPU on all native backends |
| Gradient-based text/speech training | Existing PyTorch CPU/CUDA/ROCm implementation |
| Audio feature extraction, audio input projection, acoustic GRU/vocoder | Existing Python/PyTorch implementation |
| Conductance simulator (`neurogen simulate`), STDP, structural plasticity | Existing NumPy implementation |
| Checkpoint loading, token sampling, streaming, GUI | Existing Python frontend |

Vulkan is **inference-only**, not a PyTorch device or a replacement for autograd.
The same trained `.pt` files can be loaded with any runtime backend. Native
training is rejected explicitly. Existing Python execution remains available as
the training implementation and independent reference. Continuous idle dynamics
are preserved; this change does not establish consciousness or human-like cognition.

## Build

Linux is the validated build target. CPU requires a C++17 compiler and CMake 3.20+:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
python -m neurogen language doctor --device native-cpu
```

Install the existing Python language dependencies as described in LANGUAGE.md.
The C++ shared library itself has no Python/PyTorch dependency. `make native`
builds it; `make native-test` also runs the reference parity tests.

For Vulkan, install Vulkan development headers/loader, `glslangValidator`, and
your GPU's Vulkan compute driver. On Ubuntu the build packages are
`libvulkan-dev glslang-tools`; Arch packages are `vulkan-headers vulkan-icd-loader
glslang`, plus the appropriate driver. Then:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DNEUROGEN_VULKAN=ON
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
NEUROGEN_TEST_BACKENDS=native-cpu,vulkan python -m unittest discover -s tests -p test_native.py -v
python -m neurogen language doctor --device vulkan
```

SPIR-V is compiled and embedded at build time; no external shader path is needed
at runtime. The backend selects a compute-capable GPU before a software device.
Set `NEUROGEN_VULKAN_DEVICE=0` (enumeration index) to explicitly select a device.
The doctor prints its actual device name and marks CPU Vulkan implementations
as software. A software pass is not a hardware acceleration benchmark. Vulkan
uses host-visible coherent storage buffers, explicit compute-to-host visibility,
and fences before readback. Missing support, device limits, shader/build errors,
and unavailable backends cause errors; there is no silent CPU fallback.

For NVIDIA CUDA, install the CUDA toolkit and a compatible host compiler:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DNEUROGEN_CUDA=ON
cmake --build build/native --parallel 2
NEUROGEN_TEST_BACKENDS=native-cpu,native-cuda python -m unittest discover -s tests -p test_native.py -v
python -m neurogen language doctor --device native-cuda
```

Both GPU flags may be enabled together. CUDA architecture selection can be
specified with `-DCMAKE_CUDA_ARCHITECTURES=...`. Switching a CMake option back off
requires explicitly setting it to `OFF` or using a fresh build directory.
`NEUROGEN_NATIVE_LIBRARY=/absolute/path/libneurogen_native.so` selects a library
outside the default `build/native` directory, including an installed library.

## Use existing checkpoints and continuous sessions

```bash
python -m neurogen language chat --checkpoint runs/trained/model.pt --device vulkan
python -m neurogen language stream --checkpoint runs/trained/model.pt --device native-cpu \
  --idle-hz 10 --save-session runs/session.pt
```

In the workbench, select `native-cpu`, `native-cuda`, or `vulkan` and reload the
model. The same options work for `listen` and `speak`; their spiking core is native
while audio-specific modules remain on CPU. Select `cpu`, `cuda`, `rocm`, or
`auto` to train. Reload the resulting checkpoint for native inference.

Session tensors stay on CPU in the adapter even when compute runs on a GPU.
The C++ handle holds weights and GPU resources; recurrent state is passed in and
returned independently for each invocation. This allows independent sessions,
save/resume across backends, and serialized GPU access from speech/text threads.
Native weights are an immutable snapshot. `load_state_dict` rebuilds that snapshot;
in-place tensor changes are rejected at the next call. Changes to fixed anatomy
buffers are rejected rather than silently changing the native equations.

For direct C++ embedding, include `engine/NeuralEngine.h` and link
`neurogen_native`. `NeuralEngine::step` performs token/idle processing;
`advance` accepts a width-sized input and explicit column state. The C ABI is used
by the Python adapter. Parameter serialization order is defined by
`packed_weights` and `CorticalColumnV2::load`; it is an in-process interface, not a
new disk checkpoint format. Exceptions are converted to error codes and a
thread-local diagnostic at the C boundary.

## Validation and limitations

`tests/test_native.py` compares independently implemented PyTorch and native
outputs, every recurrent state field, activity, multiple columns and batches,
idle steps, session continuation, speech frontends, parameter reload, and threaded
sessions. The C++ smoke test compares each enabled backend to native CPU. CI runs
CPU and software Vulkan parity and separately compiles CUDA.

Local validation for this change: the existing suite completed 41 tests (9
optional dependency/hardware skips); all six new parity/integration tests passed
with native CPU and llvmpipe Vulkan; both CTest backend checks passed. The Vulkan
doctor's maximum forward/state error was 9.54e-7 against PyTorch. CUDA was not
compiled or executed locally, and no physical GPU or BC-250 was available.
The CUDA compile CI job and target-device doctor are required before claiming
CUDA deployment validation.

The initial backend favors inspectability and portability over peak throughput.
GPU storage allocations are reused, but weights and state are packed/transferred
for individual operations, with synchronous readback. Dense matvec uses one
invocation per output row, batches run sequentially, and checkpoints retain CPU
parameter copies. Small models can be faster on CPU. Do not infer BC-250 speed,
VRAM capacity, or driver compatibility from this implementation: run the doctor
and benchmark your actual profile on that board before deployment.

Floating-point summation differs between implementations. Threshold crossings
can amplify small differences over long trajectories; parity tolerances are for
controlled fixtures, not a guarantee of bitwise-identical indefinite sessions.

Vulkan synchronization follows the Khronos examples:
https://docs.vulkan.org/guide/latest/synchronization_examples.html
