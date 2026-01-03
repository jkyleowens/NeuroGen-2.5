# Cortical Column V2 Implementation Status

## Overview

This document tracks the implementation progress of the new cortical column architecture for NeuroGen 2.0 based on the Dynamic Multimode Module (D3M) model.

## Completed Files

### Headers (include/engine/)

| File | Description | Status |
|------|-------------|--------|
| `ALIFNeuron.h` | ALIF neuron model with SoA layout | ✅ Complete |
| `SparseSynapseMatrix.h` | CSR/BSR sparse synapse formats | ✅ Complete |
| `CorticalColumnV2.h` | 6-layer cortical column class | ✅ Complete |
| `ALIFKernels.cuh` | CUDA kernels for ALIF updates | ✅ Complete |
| `SparseKernels.cuh` | CUDA kernels for sparse SpMV | ✅ Complete |
| `STDPKernels.cuh` | STDP learning kernels | ✅ Complete |
| `ConnectivityGenerator.h` | Connectivity pattern generators | ✅ Complete |

### Implementations (src/engine/)

| File | Description | Status |
|------|-------------|--------|
| `CorticalColumnV2.cu` | Cortical column implementation | ✅ Complete |
| `ConnectivityGenerator.cu` | Connectivity generation | ✅ Complete |
| `ALIFNeuron.cu` | ALIF neuron array methods | ✅ Complete |
| `test_cortical_column.cu` | Test suite | ✅ Complete |

## Architecture Features

### ALIF Neuron Model
- **Memory:** 44 bytes/neuron (vs 128 bytes for Izhikevich)
- **FLOPs:** ~5 per timestep (vs 40+ for Izhikevich)
- **Layout:** Structure of Arrays (SoA) for coalesced GPU memory access
- **Features:**
  - Membrane potential with exponential decay
  - Adaptive threshold (spike frequency adaptation)
  - Refractory periods
  - Activity traces for STDP

### Compartmental ALIF (for pyramidal cells)
- 3 compartments: soma, basal dendrite, apical dendrite
- Basal: feedforward input (bottom-up)
- Apical: feedback input (top-down context)
- Calcium dynamics for plasticity

### Sparse Synapse Matrix
- **CSR format:** Standard compressed sparse row
- **BSR format:** Block sparse row (32x32 blocks) for warp-level optimization
- **Features:**
  - Eligibility traces for RL
  - Pre/post synaptic activity traces
  - Dynamic connectivity support

### Cortical Column Structure
```
L1: Feedback integration (apical dendrites)
     ↑ From higher areas
     
L2/3: Cortico-cortical communication
       ↔ Lateral connections
       
L4: Thalamic input (feedforward)
     ↑ From thalamus/lower areas
     
L5: Output to subcortical structures
     ↓ To motor, subcortical
     
L6: Thalamic feedback
     ↓ To thalamus
```

### STDP Learning Rules
1. **Classic STDP:** Pair-based asymmetric
2. **Triplet STDP:** Frequency-dependent (Pfister & Gerstner)
3. **Eligibility-trace STDP:** For reinforcement learning
4. **Calcium-gated plasticity:** Threshold-based LTP/LTD
5. **Voltage-dependent STDP:** Based on Clopath et al.

### Connectivity Patterns
- Random (Erdős–Rényi)
- Small-world (Watts-Strogatz)
- Distance-dependent (Gaussian decay)
- Layer-specific (canonical cortical)

## Build Instructions

```bash
# Build everything including new cortical column code
make clean
make all

# Build and run cortical column tests
make test_cortical
./test_cortical_column
```

## Performance Targets

| Metric | Old (Izhikevich) | New (ALIF) | Improvement |
|--------|------------------|------------|-------------|
| Memory/neuron | 128 bytes | 44 bytes | 2.9x |
| FLOPs/neuron | 40+ | 5 | 8x |
| Synapse memory | 40 bytes | 8 bytes | 5x |

## Remaining Work

1. **Integration with BrainOrchestrator**
   - Create bridge between old and new architectures
   - Migration path for existing checkpoints

2. **Python Bindings**
   - Expose CorticalColumnV2 to Python
   - Add training interface

3. **D3M Generative Loop**
   - Implement bidirectional processing
   - Perception mode (bottom-up)
   - Imagination mode (top-down)

4. **Validation**
   - Test biological plausibility
   - Compare learning performance
   - Memory usage verification

## Usage Example

```cpp
#include "engine/CorticalColumnV2.h"

using namespace neurogen::cortical;

// Create column
CorticalColumnConfig config;
config.name = "V1_Column";
config.total_neurons = 10000;
config.enable_stdp = true;

CorticalColumnV2 column(config);
column.initialize();
column.generateConnectivity();

// Run simulation
float dt = 1.0f;  // 1ms
for (int t = 0; t < 1000; ++t) {
    column.injectFeedforwardInput(input_data, input_size);
    column.step(dt, t * dt);
    
    if (t % 100 == 0) {
        column.applySTDP(reward_signal);
    }
}

// Get output from L5
column.getOutput(output_buffer, output_size);
```

## Date: November 29, 2025
