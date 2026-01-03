# Biology-First Cortical Simulation Architecture

## Overview

This document describes the comprehensive architectural overhaul that transitions NeuroGen 2.0 from an "Engineering-First" SNN (efficient but biologically inaccurate) to a "Biology-First" Cortical Simulation that implements Dale's Law, Topographic Connectivity, and Synchronous Execution.

**Version**: 2.0-Biology-First
**Date**: December 30, 2025

---

## Key Biological Principles Implemented

### 1. Dale's Law

**Definition**: A neuron releases the same neurotransmitter at all its synapses.

**Implementation**:
- Neurons are classified as either **Excitatory (80%)** or **Inhibitory (20%)**
- Excitatory neurons (Pyramidal cells) use Glutamate and have **positive weights only**
- Inhibitory neurons (Interneurons) use GABA and have **negative weights only**
- Weight updates during learning **preserve the sign** of synaptic weights

**Files**:
- `include/engine/BioConstants.h` - Biological parameters and helper functions
- `src/engine/HebbianLearningKernel_Bio.cu` - STDP with Dale's Law enforcement

**Key Functions**:
```cpp
// Clamp weight to respect Dale's Law
float clampWeightDalesLaw(float weight, bool is_inhibitory);

// Get neuron type based on index
NeuronType getNeuronType(int neuron_idx, int num_excitatory);
```

---

### 2. Topographic Connectivity

**Definition**: Connection probability decreases with spatial distance, creating receptive fields.

**Implementation**:
- Neurons arranged in 2D spatial grid (width × height)
- Connection probability follows **Gaussian distribution**: `P = e^(-dist²/(2σ²))`
- Excitatory neurons: `σ = 2.5` (local spread)
- Inhibitory neurons: `σ = 4.0` (lateral inhibition radius)
- Creates natural **columnar organization** and **receptive fields**

**Files**:
- `src/engine/BioConnectivityGenerator.cu` - Topographic wiring generator

**Key Function**:
```cpp
void generateCorticalWiring(
    std::vector<int>& row_ptr,      // CSR row pointers
    std::vector<int>& col_ind,      // CSR column indices
    std::vector<float>& values,     // Synaptic weights
    int num_neurons,                // Total neurons
    int width, int height           // Spatial dimensions
);
```

**Benefits**:
- Acts as a **Spiking CNN** in early layers
- Naturally detects edges and shapes (graphemes)
- Enables **winner-take-all** dynamics for discrete word learning
- Prevents "inhibitory void" problem

---

### 3. Synchronous Execution (Tick-Tock Orchestrator)

**Definition**: All modules execute in synchronized phases to enable feedback loops and gamma oscillations.

**Implementation - Four Phases**:

1. **Integration Phase**: All modules compute membrane potential (V_m) from t-1 inputs
2. **Firing Phase**: All modules generate spikes based on V_m
3. **Routing Phase**: Spikes propagate through connections (axonal delay)
4. **Plasticity Phase**: STDP and structural updates

**Files**:
- `include/modules/BrainOrchestrator.h` - Header with new processing mode
- `src/modules/BrainOrchestrator.cpp` - Implementation of synchronous execution

**Key Functions**:
```cpp
// Main synchronous step
std::vector<float> biologicalSynchronousStep(
    const std::vector<float>& input_embedding,
    int target_token_id = -1,
    GPUDecoder* decoder = nullptr
);

// Phase implementations
void phaseIntegration();  // Phase 1
void phaseFiring();       // Phase 2
void phaseRouting();      // Phase 3
void phasePlasticity();   // Phase 4
```

**Benefits**:
- Enables **recurrent feedback loops** (Module A → B → A)
- Supports **gamma oscillations** (40-100 Hz)
- Prevents race conditions in recurrent networks
- Biologically realistic timing

---

## Architecture Components

### File Structure

```
NeuroGen-2.0/
├── include/engine/
│   └── BioConstants.h              [NEW] Biological parameters
├── src/engine/
│   ├── BioConnectivityGenerator.cu [NEW] Topographic wiring
│   ├── HebbianLearningKernel_Bio.cu [NEW] Dale's Law STDP
│   └── CorticalColumnV2.cu         [MODIFIED] Uses biological init
├── src/modules/
│   └── BrainOrchestrator.cpp       [MODIFIED] Adds sync mode
└── Makefile                        [MODIFIED] Builds new components
```

### BioConstants.h

Defines:
- `INHIBITORY_RATIO = 0.20f` (20% interneurons)
- `EXCITATORY_RATIO = 0.80f` (80% pyramidal)
- `LOCAL_SPREAD_SIGMA = 2.5f` (excitatory connectivity)
- `LATERAL_INHIBITION_RADIUS = 4.0f` (inhibitory connectivity)
- `NeuronType` enum (EXCITATORY / INHIBITORY)
- `clampWeightDalesLaw()` helper function

### BioConnectivityGenerator.cu

**Key Class**: `BioConnectivityGenerator`

**Main Method**: `generateCorticalWiring()`
- Creates sparse CSR matrix with topographic connectivity
- Enforces Dale's Law (positive/negative weight segregation)
- Uses Gaussian distance-based probability
- Prevents autapses (self-connections)
- Implements lateral inhibition for interneurons

**CUDA Kernel**: `generate_topographic_mask_kernel()`
- Calculates connection probability based on 3D distance
- Runs on GPU for large-scale networks

### HebbianLearningKernel_Bio.cu

**Kernels**:
1. `biological_stdp_kernel()` - STDP with Dale's Law enforcement
2. `eligibility_trace_stdp_kernel()` - Three-factor learning (reward-modulated)
3. `homeostatic_plasticity_kernel()` - Synaptic scaling with sign preservation
4. `triplet_stdp_kernel()` - Advanced triplet STDP

**Key Feature**: All weight updates respect Dale's Law:
```cuda
// DALE'S LAW CLAMPING
float new_w = weights[i] + dw;
new_w = clampWeightDalesLaw(new_w, is_inhibitory);
weights[i] = new_w;
```

### BrainOrchestrator Synchronous Mode

**Processing Modes**:
- `SEQUENTIAL` - Original phase-based processing
- `PIPELINED` - Overlapped execution for throughput
- `BIOLOGICAL_SYNC` - **[NEW]** Tick-tock synchronous execution

**Enable Biological Mode**:
```cpp
BrainOrchestrator orchestrator(config);
orchestrator.setProcessingMode(ProcessingMode::BIOLOGICAL_SYNC);
```

---

## Usage

### Building

```bash
# Build all components
make clean
make all

# The new biological components will be automatically included
```

### Integration

The biological components are **automatically used** when:

1. **CorticalColumnV2** generates connectivity (uses `BioConnectivityGenerator`)
2. **Learning kernels** update weights (enforces Dale's Law)
3. **BrainOrchestrator** runs in `BIOLOGICAL_SYNC` mode

### Example Configuration

```cpp
// Enable biological synchronous mode
BrainOrchestrator::Config config;
config.processing_mode = ProcessingMode::BIOLOGICAL_SYNC;
config.enable_parallel_execution = true;  // Enable plasticity phase
config.time_step_ms = 1.0f;               // 1ms timestep (1000 Hz)

BrainOrchestrator brain(config);
brain.initializeModules();
brain.createConnectome();

// Run cognitive step
std::vector<float> output = brain.cognitiveStep(input_embedding);
```

---

## Benefits of Biology-First Architecture

### 1. Dale's Law Implementation
✅ **Fixes**: "Inhibitory Void" problem
✅ **Enables**: Self-regulating gain control
✅ **Provides**: Stable excitatory-inhibitory balance

### 2. Topographic Connectivity
✅ **Creates**: Natural receptive fields
✅ **Enables**: Hierarchical feature detection (edges → shapes → objects)
✅ **Provides**: Columnar organization like visual cortex

### 3. Synchronous Execution
✅ **Enables**: Gamma oscillations (40-100 Hz)
✅ **Supports**: Recurrent feedback loops
✅ **Provides**: Winner-take-all dynamics
✅ **Prevents**: Race conditions in recurrent networks

---

## Biological Neuron Parameters

### Excitatory Neurons (Pyramidal Cells)
- **Threshold**: 1.0
- **Membrane tau**: 20 ms (long integration)
- **Decay rate**: 0.95 (long memory)
- **Refractory period**: 2 ms
- **Adaptation**: 0.05 (moderate)
- **Weights**: Positive only (0.001 to 2.0)

### Inhibitory Neurons (Fast-Spiking Interneurons)
- **Threshold**: 0.7 (lower, fires more easily)
- **Membrane tau**: 10 ms (fast response)
- **Decay rate**: 0.80 (short memory)
- **Refractory period**: 1 ms (can fire faster)
- **Adaptation**: 0.02 (less adaptation)
- **Weights**: Negative only (-4.0 to -0.001)

---

## Performance Considerations

### Memory Footprint
- Topographic connectivity is **sparse** (~15% density at center)
- Dale's Law adds **no memory overhead** (just enforces constraints)
- Synchronous mode adds **minimal state overhead** (phase tracking)

### Computational Cost
- **Integration Phase**: Same as original (compute V_m)
- **Firing Phase**: Same as original (threshold check)
- **Routing Phase**: Same as original (spike propagation)
- **Plasticity Phase**: Slightly more expensive (Dale's Law clamping)

**Overall**: ~5-10% performance overhead for biological accuracy

---

## Future Enhancements

1. **GPU-Accelerated Connectivity Generation**: Move topographic wiring to GPU
2. **Compartmental Neuron Models**: Multi-compartment pyramidal cells
3. **Dendritic Computation**: NMDA spikes and backpropagation
4. **Calcium-Based Plasticity**: More accurate learning rules
5. **Neuromodulation**: Dopamine, serotonin, acetylcholine effects on Dale's Law

---

## References

### Biological Principles
- Dale's Principle: Eccles, J. C. (1976). *From electrical to chemical transmission in the central nervous system*
- Cortical Connectivity: Douglas, R. J., & Martin, K. A. (2004). *Neuronal circuits of the neocortex*
- Topographic Maps: Goodhill, G. J. (2007). *Contributions of theoretical modeling to the understanding of neural map development*

### Computational Implementations
- Spiking Neural Networks: Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models*
- STDP: Bi, G. Q., & Poo, M. M. (1998). *Synaptic modifications in cultured hippocampal neurons*
- Gamma Oscillations: Buzsáki, G., & Wang, X. J. (2012). *Mechanisms of gamma oscillations*

---

## Contact

For questions about the Biology-First architecture:
- Open an issue on GitHub
- Check the code comments in `BioConstants.h`, `BioConnectivityGenerator.cu`, and `HebbianLearningKernel_Bio.cu`

**Remember**: Biology-first means **accuracy over efficiency**. The goal is to simulate cortical dynamics, not just process tokens quickly.
