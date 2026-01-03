# NeuroGen 2.0 Cortical Column Architecture - Overhaul Plan

**Version:** 2.0-Cortical  
**Date:** November 29, 2025  
**Status:** Design Specification  
**Objective:** Transform NeuroGen from modular brain regions to biologically-inspired cortical column architecture

---

## Executive Summary

This document outlines the complete architectural transformation of NeuroGen 2.0 from its current "regional module" design (Thalamus, Wernicke, Broca, etc.) to a **cortical column-based neuromorphic architecture** inspired by the Dynamic Multimode Module (D3M) model and Linear Attention Transformers (RWKV/SpikeGPT).

### Key Changes:
1. **Neuron Model**: Transition from Izhikevich → Adaptive LIF (ALIF)
2. **Synapse Model**: Transition from individual structs → Sparse Matrix (CSR/BSR)
3. **Module Structure**: Transition from Brain Regions → Cortical Columns (6-layer structure)
4. **Learning**: Enhance STDP with compartment-specific calcium dynamics
5. **Architecture**: Implement D3M bidirectional generative loops
6. **Topology**: Small-world network with Rich Club hubs

---

## Part 1: Neuron Model Transformation

### 1.1 Current State Analysis

**Current Neuron (Izhikevich-based):**
```cpp
struct GPUNeuronState {
    float voltage;              // Membrane potential
    float recovery;             // Recovery variable (u)
    float ca_conc[4];          // Calcium per compartment
    float last_spike_time;
    float spike_threshold;
    // ~128 bytes per neuron in AoS layout
};
```

**Problems:**
- Requires multiple sub-steps per timestep (numerical integration overhead)
- Memory access patterns cause cache misses (AoS layout)
- Complex dynamics limit scaling to millions of neurons

### 1.2 New Neuron Model: Adaptive LIF (ALIF)

**Design Philosophy:** Preserve essential spiking dynamics while maximizing computational efficiency.

**State Variables (SoA Layout):**
```cpp
// Structure of Arrays - each is contiguous float array
struct ALIFNeuronState {
    float* d_voltage;          // V[i] - Membrane potential
    float* d_adaptation;       // A[i] - Spike-triggered adaptation
    float* d_threshold;        // Θ[i] - Dynamic spike threshold
    float* d_ca_conc;          // Ca[i] - Compartment calcium (simplified to 1 per neuron initially)
    uint8_t* d_spike_flags;    // Binary spike indicators
    float* d_last_spike_time;  // For STDP timing
    // ~32 bytes per neuron in SoA layout
};
```

**Dynamics (Discrete-Time):**
```cuda
__global__ void alif_update_kernel(
    float* __restrict__ voltage,
    float* __restrict__ adaptation,
    float* __restrict__ threshold,
    const float* __restrict__ input_current,
    uint8_t* __restrict__ spikes,
    float* __restrict__ last_spike_time,
    int num_neurons,
    float dt,
    float current_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Parameters
    const float alpha = 0.95f;     // Voltage decay (τ_m = 20ms → α = exp(-dt/τ))
    const float beta = 0.90f;      // Adaptation decay (τ_a = 100ms)
    const float delta = 1.5f;      // Adaptation increment per spike
    const float V_rest = -65.0f;
    const float V_reset = -70.0f;
    const float Θ_base = -50.0f;
    
    // Load state
    float V = voltage[idx];
    float A = adaptation[idx];
    float Θ = threshold[idx];
    float I = input_current[idx];
    
    // Update voltage: V_{t+1} = α*V_t + I_t - A_t
    V = alpha * (V - V_rest) + V_rest + I - A;
    
    // Spike detection
    bool spiked = (V > Θ);
    spikes[idx] = spiked ? 1 : 0;
    
    if (spiked) {
        V = V_reset;
        A += delta;
        threshold[idx] = Θ_base + 5.0f; // Temporary threshold increase
        last_spike_time[idx] = current_time;
    } else {
        // Decay threshold back to baseline
        threshold[idx] = beta * Θ + (1.0f - beta) * Θ_base;
    }
    
    // Decay adaptation
    A *= beta;
    
    // Store state
    voltage[idx] = V;
    adaptation[idx] = A;
}
```

**Benefits:**
- 1 FMA per neuron per step (vs. 6-10 FLOPs in Izhikevich)
- Coalesced memory access (SoA)
- 4x memory reduction
- Preserves: adaptation, refractory periods, dynamic thresholds

### 1.3 Multi-Compartment Extension (Future)

For cortical columns requiring dendritic computation:

```cpp
struct CompartmentalALIF {
    float* d_soma_voltage;      // Soma
    float* d_basal_voltage;     // Basal dendrites (L2/3 input)
    float* d_apical_voltage;    // Apical dendrites (L1 feedback)
    float* d_adaptation;
    float* d_threshold;
};
```

**Implementation:** Couple compartments with conductance terms:
```
V_soma' = α*V_soma + g_basal*(V_basal - V_soma) + g_apical*(V_apical - V_soma) - A
```

---

## Part 2: Synapse Model Transformation

### 2.1 Current State Analysis

**Current Synapse:**
```cpp
struct GPUSynapse {
    int pre_neuron_idx;
    int post_neuron_idx;
    float weight;
    float eligibility_trace;
    float learning_rate;
    float dopamine_sensitivity;
    int post_compartment;  // Target compartment
    // ~40 bytes per synapse
};
```

**Problems:**
- Random memory access (pointer chasing)
- Inefficient parallel processing
- Cannot utilize tensor cores
- Difficult to prune/grow dynamically

### 2.2 New Synapse Model: Block Sparse Row (BSR) Matrix

**Design Philosophy:** Treat synaptic connectivity as sparse matrix operations.

**Data Structure:**
```cpp
struct SparseWeightMatrix {
    // BSR format with block size 32x32 (warp-optimized)
    float* d_weight_values;      // [num_blocks * 1024] - Dense 32x32 weight blocks
    int* d_col_indices;          // [num_blocks] - Column index of each block
    int* d_row_ptr;              // [num_neurons/32 + 1] - Row pointer array
    
    // Plasticity tracking (parallel arrays)
    float* d_eligibility_trace;  // [num_blocks * 1024]
    float* d_dopamine_sensitivity; // [num_neurons] - per-neuron modulation
    
    int num_blocks;
    int block_size;              // 32
};
```

**Synaptic Propagation (SpMV Kernel):**
```cuda
__global__ void sparse_synaptic_propagation_kernel(
    const float* __restrict__ weight_values,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_ptr,
    const uint8_t* __restrict__ pre_spikes,  // Sparse binary vector
    float* __restrict__ post_currents,       // Output currents
    int num_neuron_blocks,
    int block_size
) {
    // Each block processes one row-block (32 post-neurons)
    int row_block = blockIdx.x;
    if (row_block >= num_neuron_blocks) return;
    
    int tid = threadIdx.x; // 0-31 (warp)
    int post_neuron = row_block * block_size + tid;
    
    float accumulated_current = 0.0f;
    
    // Iterate over non-zero blocks in this row
    int block_start = row_ptr[row_block];
    int block_end = row_ptr[row_block + 1];
    
    for (int b = block_start; b < block_end; ++b) {
        int col_block = col_indices[b];
        int pre_neuron_base = col_block * block_size;
        
        // Load 32x32 weight block into shared memory
        __shared__ float weights[32][32];
        weights[tid][0] = weight_values[b * 1024 + tid];
        // ... load full block ...
        __syncthreads();
        
        // Dot product with pre-synaptic spike vector
        for (int j = 0; j < block_size; ++j) {
            int pre_idx = pre_neuron_base + j;
            if (pre_spikes[pre_idx]) {
                accumulated_current += weights[tid][j];
            }
        }
        __syncthreads();
    }
    
    // Atomic add to handle multiple blocks targeting same post-neuron
    atomicAdd(&post_currents[post_neuron], accumulated_current);
}
```

**Benefits:**
- Leverage GPU tensor cores (future: use WMMA for FP16)
- Coalesced memory access (blocks loaded contiguously)
- 10-100x faster than individual synapse loops
- Natural support for pruning (remove sparse blocks)

### 2.3 Plasticity in Matrix Form

**Three-Factor STDP Update:**
```cuda
__global__ void matrix_stdp_update_kernel(
    float* __restrict__ weight_values,
    float* __restrict__ eligibility_trace,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_ptr,
    const uint8_t* __restrict__ pre_spikes,
    const uint8_t* __restrict__ post_spikes,
    const float* __restrict__ pre_spike_times,
    const float* __restrict__ post_spike_times,
    float dopamine_signal,
    float current_time,
    float dt,
    int num_blocks,
    int block_size
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;
    
    // Process each synapse in the block
    for (int i = 0; i < block_size * block_size; ++i) {
        int syn_idx = block_idx * 1024 + i;
        int post_local = i / block_size;
        int pre_local = i % block_size;
        
        int row_block = findRowBlock(block_idx, row_ptr);
        int col_block = col_indices[block_idx];
        
        int post_neuron = row_block * block_size + post_local;
        int pre_neuron = col_block * block_size + pre_local;
        
        // Calculate spike timing
        float delta_t = post_spike_times[post_neuron] - pre_spike_times[pre_neuron];
        
        // STDP window
        float stdp_magnitude = 0.0f;
        if (fabsf(delta_t) < 20.0f) {
            if (delta_t > 0) {
                stdp_magnitude = __expf(-delta_t / 10.0f);  // LTP
            } else {
                stdp_magnitude = -0.8f * __expf(delta_t / 10.0f);  // LTD
            }
        }
        
        // Update eligibility trace
        eligibility_trace[syn_idx] *= 0.95f;  // Decay
        eligibility_trace[syn_idx] += stdp_magnitude;
        
        // Three-factor rule: ΔW = η * dopamine * eligibility
        float weight_change = 0.01f * dopamine_signal * eligibility_trace[syn_idx] * dt;
        
        // Update weight
        weight_values[syn_idx] += weight_change;
        weight_values[syn_idx] = fmaxf(0.0f, fminf(2.0f, weight_values[syn_idx]));
    }
}
```

---

## Part 3: Cortical Column Structure

### 3.1 The Neuromorphic Generative Module (NGM)

Replace brain region modules (Thalamus, Wernicke, Broca) with **uniform cortical columns** implementing the 6-layer D3M architecture.

**Column Structure:**
```cpp
struct CorticalColumn {
    // Layer assignments (indices into neuron arrays)
    struct LayerIndices {
        int layer1_start, layer1_end;    // Context Integration (5% of neurons)
        int layer2_3_start, layer2_3_end; // Associative Memory (35%)
        int layer4_start, layer4_end;     // Sensory Input (20%)
        int layer5_start, layer5_end;     // Generative Output (25%)
        int layer6_start, layer6_end;     // Feedback Control (15%)
    } layers;
    
    // Intra-column connectivity (sparse matrices)
    SparseWeightMatrix L4_to_L23;      // Bottom-up perception
    SparseWeightMatrix L23_recurrent;  // Lateral associations (Key-Value)
    SparseWeightMatrix L5_to_L23;      // Top-down query (D3M backward)
    SparseWeightMatrix L23_to_L5;      // Context to output
    SparseWeightMatrix L1_to_L5;       // Global context → Apical dendrites
    SparseWeightMatrix L6_to_TRN;      // Attention gating
    
    // Column metadata
    int column_id;
    int total_neurons;
    ColumnType type;  // SENSORY, ASSOCIATIVE, MOTOR
};

enum class ColumnType {
    SENSORY,      // Like old Thalamus/Wernicke (bottom-up)
    ASSOCIATIVE,  // Like old Hippocampus/PFC (integration)
    MOTOR         // Like old Broca/Basal Ganglia (top-down)
};
```

### 3.2 Layer-Specific Dynamics

**Layer 4: Prediction Error Unit**
```cuda
// Calculate prediction error: Actual - Predicted
__global__ void layer4_prediction_error(
    const float* __restrict__ sensory_input,    // From thalamus
    const float* __restrict__ layer6_prediction, // From L6 feedback
    float* __restrict__ layer4_activity,
    int num_layer4_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_layer4_neurons) return;
    
    float error = sensory_input[idx] - layer6_prediction[idx];
    layer4_activity[idx] = fmaxf(0.0f, error); // Rectified error
}
```

**Layer 2/3: Recurrent Associative Memory (RWKV-style)**
```cuda
// Linear attention with running state
__global__ void layer23_rwkv_update(
    float* __restrict__ state,          // Running KV state
    const float* __restrict__ layer4_input,
    float* __restrict__ layer23_output,
    const float* __restrict__ receptance_weights,
    float decay_factor,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // State update: S_t = decay * S_{t-1} + Input_t
    state[idx] = decay_factor * state[idx] + layer4_input[idx];
    
    // Output: O_t = sigmoid(Receptance) * State_t
    float receptance = receptance_weights[idx];
    float gate = 1.0f / (1.0f + __expf(-receptance));
    layer23_output[idx] = gate * state[idx];
}
```

**Layer 5: Query Generation & Bursting**
```cuda
// Coincidence detection for burst generation
__global__ void layer5_burst_detection(
    const float* __restrict__ basal_input,   // From L2/3
    const float* __restrict__ apical_input,  // From L1 (global context)
    uint8_t* __restrict__ burst_flags,
    float* __restrict__ layer5_output,
    float coincidence_threshold,
    int num_layer5_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_layer5_neurons) return;
    
    float basal = basal_input[idx];
    float apical = apical_input[idx];
    
    // Burst if both basal AND apical are active (AND gate)
    bool burst = (basal > coincidence_threshold) && (apical > coincidence_threshold);
    
    burst_flags[idx] = burst ? 1 : 0;
    layer5_output[idx] = burst ? 2.0f * basal : 0.0f; // Amplified output
}
```

**Layer 6: Precision Weighting (Attention)**
```cuda
// Calculate confidence/precision for gating
__global__ void layer6_precision_weighting(
    const float* __restrict__ layer23_state,
    float* __restrict__ layer6_confidence,
    float* __restrict__ trn_inhibition,  // Output to TRN
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Confidence = inverse variance of predictions
    float prediction_variance = 0.1f; // Simplified
    layer6_confidence[idx] = 1.0f / (prediction_variance + 1e-6f);
    
    // High confidence → Inhibit TRN → Open gate
    // Low confidence → Disinhibit TRN → Close gate
    trn_inhibition[idx] = layer6_confidence[idx];
}
```

### 3.3 D3M Generative Loop Implementation

**Bottom-Up (Perception Mode):**
```
Input → L4 (Error) → L2/3 (Association) → L5 (Output)
```

**Top-Down (Generation Mode):**
```
L5 (Query) → L2/3 (Retrieval) → L6 (Feedback) → Thalamus/L4
```

```cpp
void CorticalColumn::update(float dt, OperatingMode mode) {
    if (mode == OperatingMode::PERCEPTION) {
        // Forward sweep
        propagate_L4_to_L23();
        update_L23_recurrent();
        propagate_L23_to_L5();
    } else if (mode == OperatingMode::GENERATION) {
        // Backward sweep (D3M)
        generate_L5_query();
        retrieve_L23_from_query();
        feedback_L6_to_L4();
        broadcast_L5_to_L1_global();
    }
}
```

---

## Part 4: Network Topology & Inter-Column Connectivity

### 4.1 Small-World Architecture

**Design Principles:**
1. **Local Clustering**: Columns within macro-regions densely connected (80%)
2. **Long-Range Shortcuts**: Sparse connections between distant regions (5-10%)
3. **Rich Club Hubs**: Highly connected "integration" columns

**Implementation:**
```cpp
struct NetworkTopology {
    std::vector<CorticalColumn> columns;
    
    // Macro-column groupings (like old brain regions)
    struct MacroColumn {
        std::vector<int> column_indices;
        MacroColumnType type;  // SENSORY, SEMANTIC, EXECUTIVE, MOTOR
    };
    std::vector<MacroColumn> macro_columns;
    
    // Inter-column connectivity
    SparseWeightMatrix long_range_connections;  // Between macro-columns
    std::vector<int> rich_club_hubs;           // High-degree nodes
};
```

**Connection Probability:**
```cpp
float connection_probability(int col_i, int col_j) {
    float distance = spatial_distance(col_i, col_j);
    
    // Distance-dependent probability
    if (distance < local_radius) {
        return 0.8f;  // High local connectivity
    } else {
        // Long-range: exponential decay + rich club boost
        float p_base = 0.05f * exp(-distance / length_scale);
        
        bool is_hub_pair = is_rich_club(col_i) && is_rich_club(col_j);
        return is_hub_pair ? p_base * 5.0f : p_base;
    }
}
```

### 4.2 Mapping Old Modules to Column Types

| Old Module | New Implementation | Column Count | Type |
|------------|-------------------|--------------|------|
| Thalamus | Sensory Column Cluster | 4-8 columns | SENSORY |
| Wernicke | Semantic Association Columns | 16-32 columns | ASSOCIATIVE |
| Hippocampus | Fast-Learning Columns | 8-16 columns | ASSOCIATIVE |
| PFC | Executive Integration Columns | 32-64 columns | ASSOCIATIVE |
| Basal Ganglia | Action Selection Columns | 8-16 columns | MOTOR |
| Broca | Motor Output Columns | 16-32 columns | MOTOR |

**Total Columns:** 100-200 columns × 5,000-10,000 neurons/column = 500K-2M neurons

### 4.3 Global Workspace Implementation

**Thalamic Broadcasting (Matrix Cells):**
```cuda
__global__ void thalamic_broadcast_kernel(
    const uint8_t* __restrict__ layer5_bursts,  // From all columns
    float* __restrict__ layer1_context,         // To all columns
    const int* __restrict__ column_offsets,
    float* __restrict__ global_state,
    int num_columns
) {
    int col_idx = blockIdx.x;
    if (col_idx >= num_columns) return;
    
    int tid = threadIdx.x;
    
    // Detect which column "won" (highest burst activity)
    __shared__ int winner_column;
    if (tid == 0) {
        float max_activity = 0.0f;
        for (int c = 0; c < num_columns; ++c) {
            float activity = calculate_column_burst_rate(layer5_bursts, c);
            if (activity > max_activity) {
                max_activity = activity;
                winner_column = c;
            }
        }
    }
    __syncthreads();
    
    // Broadcast winner's state to Layer 1 of all columns
    int winner_offset = column_offsets[winner_column];
    float broadcast_value = global_state[winner_offset + tid];
    
    // Write to all columns' Layer 1
    for (int c = 0; c < num_columns; ++c) {
        int layer1_offset = column_offsets[c]; // Layer 1 start
        layer1_context[layer1_offset + tid] = broadcast_value;
    }
}
```

---

## Part 5: Enhanced Learning Rules

### 5.1 Calcium-Based Compartmental STDP

**Current:** Simple spike-timing differences  
**New:** Compartment-specific calcium accumulation

```cuda
__device__ float update_calcium(
    float ca_current,
    bool pre_spiked,
    bool post_spiked,
    float dt
) {
    const float tau_ca = 50.0f;  // 50ms decay
    const float ca_increment = 0.5f;
    
    // Decay
    ca_current *= expf(-dt / tau_ca);
    
    // Increment on coincidence
    if (pre_spiked && post_spiked) {
        ca_current += ca_increment;
    }
    
    return fminf(ca_current, 2.0f);  // Saturation
}

__global__ void calcium_stdp_kernel(
    float* __restrict__ weights,
    float* __restrict__ ca_conc,
    const uint8_t* __restrict__ pre_spikes,
    const uint8_t* __restrict__ post_spikes,
    float* __restrict__ eligibility,
    float dopamine,
    float dt,
    int num_synapses
) {
    int syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (syn_idx >= num_synapses) return;
    
    // Update calcium
    float ca = update_calcium(
        ca_conc[syn_idx],
        pre_spikes[syn_idx],
        post_spikes[syn_idx],
        dt
    );
    ca_conc[syn_idx] = ca;
    
    // Plasticity magnitude depends on calcium level
    float plasticity_factor = 1.0f;
    if (ca > 1.5f) {
        plasticity_factor = 2.0f;  // Supralinear LTP
    } else if (ca < 0.5f) {
        plasticity_factor = 0.5f;  // Weak plasticity
    }
    
    // Three-factor rule with calcium modulation
    float dw = 0.01f * dopamine * eligibility[syn_idx] * plasticity_factor * dt;
    weights[syn_idx] += dw;
}
```

### 5.2 Layer-Specific Learning Rates

Different layers have different plasticity profiles:

```cpp
struct LayerPlasticityConfig {
    float learning_rate;
    float dopamine_sensitivity;
    float calcium_threshold;
};

// Layer 4: Fast adaptation to sensory statistics
LayerPlasticityConfig layer4_config = {
    .learning_rate = 0.05f,
    .dopamine_sensitivity = 0.3f,
    .calcium_threshold = 0.5f
};

// Layer 2/3: Slow semantic consolidation
LayerPlasticityConfig layer23_config = {
    .learning_rate = 0.01f,
    .dopamine_sensitivity = 0.5f,
    .calcium_threshold = 1.0f
};

// Layer 5: Reward-driven motor learning
LayerPlasticityConfig layer5_config = {
    .learning_rate = 0.03f,
    .dopamine_sensitivity = 1.0f,  // Very sensitive!
    .calcium_threshold = 0.8f
};
```

### 5.3 Pure STDP vs. Reward-Modulated Regions

**Key Architectural Decision:** Some columns use pure STDP (sensory), others use reward-modulated plasticity (motor).

```cuda
__global__ void adaptive_learning_kernel(
    float* __restrict__ weights,
    const float* __restrict__ eligibility,
    float dopamine,
    const uint8_t* __restrict__ column_learning_mode,  // 0=pure STDP, 1=reward
    int num_synapses
) {
    int syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (syn_idx >= num_synapses) return;
    
    int column_id = syn_idx / SYNAPSES_PER_COLUMN;
    uint8_t mode = column_learning_mode[column_id];
    
    float dw;
    if (mode == 0) {
        // Pure STDP (ignore dopamine)
        dw = 0.01f * eligibility[syn_idx];
    } else {
        // Reward-modulated
        dw = 0.01f * dopamine * eligibility[syn_idx];
    }
    
    weights[syn_idx] += dw;
}
```

---

## Part 6: Implementation Roadmap

### Phase 1: Core Neuron & Synapse Refactor (Weeks 1-2)

**Tasks:**
1. Implement ALIF neuron model (SoA layout)
2. Create sparse matrix data structures (CSR/BSR)
3. Write SpMV kernels for synaptic propagation
4. Benchmark against current Izhikevich implementation

**Success Criteria:**
- [ ] 10x faster neuron updates
- [ ] 5x memory reduction
- [ ] Pass unit tests for spike timing accuracy

### Phase 2: Single Column Implementation (Weeks 3-4)

**Tasks:**
1. Implement 6-layer column structure
2. Create intra-column connectivity kernels
3. Implement D3M forward/backward sweeps
4. Test with toy language task (e.g., XOR, sequence memory)

**Success Criteria:**
- [ ] Column can perform basic pattern completion
- [ ] D3M generative loop works (L5→L2/3→L6)
- [ ] STDP learns simple associations

### Phase 3: Multi-Column Network (Weeks 5-6)

**Tasks:**
1. Implement network topology builder
2. Create inter-column sparse connectivity
3. Implement global workspace broadcasting
4. Scale to 10-50 columns

**Success Criteria:**
- [ ] Network supports 100K-500K neurons
- [ ] Inter-column communication works
- [ ] Global ignition events detectable

### Phase 4: Migration from Old Architecture (Weeks 7-8)

**Tasks:**
1. Map old module weights to new column structure
2. Create compatibility layer for existing checkpoints
3. Implement progressive migration (hybrid mode)
4. Validate that language performance doesn't degrade

**Success Criteria:**
- [ ] Can load old Broca/Wernicke weights
- [ ] Perplexity on SlimPajama comparable to v1.x
- [ ] Training can resume from old checkpoints

### Phase 5: Advanced Features (Weeks 9-12)

**Tasks:**
1. Implement Active Inference loops
2. Add structural plasticity (pruning/growth)
3. Optimize for A100 GPUs (tensor cores, FP16)
4. Scale to 1M+ neurons

**Success Criteria:**
- [ ] Autonomous "thinking" mode works
- [ ] Network self-organizes connectivity
- [ ] 100x energy efficiency improvement

---

## Part 7: Code Structure Changes

### 7.1 New File Organization

```
src/
├── core/
│   ├── ALIFNeuron.cu           # New neuron model
│   ├── SparseMatrix.cu         # Sparse weight matrices
│   └── Compartments.cu         # Multi-compartment extensions
├── column/
│   ├── CorticalColumn.cu       # 6-layer column implementation
│   ├── LayerDynamics.cu        # Layer-specific kernels
│   └── D3MLoops.cu             # Generative loop logic
├── network/
│   ├── Topology.cu             # Small-world network builder
│   ├── GlobalWorkspace.cu      # Thalamic broadcasting
│   └── RichClub.cu             # Hub detection/creation
├── learning/
│   ├── CalciumSTDP.cu          # Enhanced plasticity
│   ├── ThreeFactorRule.cu      # Reward modulation
│   └── StructuralPlasticity.cu # Pruning/growth
└── migration/
    ├── LegacyLoader.cu         # Old checkpoint compatibility
    └── ModuleToColumn.cu       # Weight mapping
```

### 7.2 Key Classes

**New Core Classes:**
```cpp
class ALIFPopulation {
    ALIFNeuronState state_;
    void update(float dt);
    std::vector<float> getSpikes();
};

class SparseWeightMatrix {
    BSRData data_;
    void propagateSpikes(const uint8_t* pre_spikes, float* post_currents);
    void updateWeights(float dopamine, float dt);
};

class CorticalColumn {
    ALIFPopulation neurons_;
    std::map<std::string, SparseWeightMatrix> connections_;
    void updatePerception(float dt);
    void updateGeneration(float dt);
};

class ColumnNetwork {
    std::vector<CorticalColumn> columns_;
    SparseWeightMatrix inter_column_weights_;
    void broadcast(int winner_column);
    void update(float dt, OperatingMode mode);
};
```

---

## Part 8: Validation & Testing

### 8.1 Unit Tests

```cpp
TEST(ALIFNeuron, SpikeTiming) {
    // Verify spike timing matches biological data
    // Input: constant current → should spike at predictable intervals
}

TEST(SparseMatrix, SpMVCorrectness) {
    // Compare SpMV output to dense matrix multiplication
}

TEST(CorticalColumn, D3MGenerativeLoop) {
    // Query L5 → should retrieve from L2/3
}

TEST(NetworkTopology, SmallWorldProperties) {
    // Verify clustering coefficient and path length
}
```

### 8.2 Integration Tests

```cpp
TEST(FullNetwork, LanguageGeneration) {
    // Load SlimPajama minibatch
    // Verify perplexity < baseline
}

TEST(Migration, CheckpointCompatibility) {
    // Load old v1.x checkpoint
    // Verify weights transfer correctly
}
```

### 8.3 Performance Benchmarks

```cpp
BENCHMARK(ALIFvsIzhikevich) {
    // Compare update speed at 100K, 500K, 1M neurons
}

BENCHMARK(SpMVvsPairwiseSynapses) {
    // Compare synaptic propagation methods
}
```

---

## Part 9: Expected Performance Improvements

### 9.1 Computational Efficiency

| Metric | Current (v1.x) | Target (v2.0-Cortical) | Improvement |
|--------|----------------|------------------------|-------------|
| Neuron Update (FLOPs) | 10 FLOPs/neuron | 1 FLOP/neuron | 10x |
| Synapse Propagation | O(N_syn) serial | O(N_neurons) parallel | 50-100x |
| Memory per Neuron | 128 bytes | 32 bytes | 4x |
| Max Network Size (4GB) | ~100K neurons | ~1M neurons | 10x |
| Energy per Inference | - | ~1000x reduction (vs Transformers) | New capability |

### 9.2 Biological Realism

| Feature | Current | Target |
|---------|---------|--------|
| Cortical Layers | No | 6-layer D3M |
| Generative Loops | No | L5→L2/3→L6 |
| Attention Mechanism | Heuristic | TRN gating |
| Global Workspace | No | Thalamic broadcast |
| Active Inference | No | Latent loops |

---

## Part 10: Open Questions & Decisions Needed

### 10.1 Architecture Decisions

1. **Column Granularity**: 5K or 10K neurons per column?
   - **Recommendation**: Start with 5K, scale up once proven

2. **Block Sparse Size**: 16x16 or 32x32 weight blocks?
   - **Recommendation**: 32x32 for warp efficiency

3. **Compartment Count**: 1, 2, or 3 compartments per neuron?
   - **Recommendation**: Phase 1: 1 compartment, Phase 5: 3 compartments

4. **Learning Mode**: Pure STDP, Reward-only, or Hybrid per column?
   - **Recommendation**: Hybrid with column-type tags

### 10.2 Implementation Details

1. **Checkpoint Format**: New format or extend old?
   - **Recommendation**: New format with migration tool

2. **Python Bindings**: Expose columns or maintain module abstraction?
   - **Recommendation**: Both - columns for experts, modules for users

3. **Distributed Training**: Multi-GPU column assignment?
   - **Recommendation**: Phase 6 (post-MVP)

---

## Conclusion

This architectural overhaul transforms NeuroGen from a collection of specialized brain regions into a scalable, biologically-inspired cortical column network. The key innovations are:

1. **ALIF neurons** for 10x computational efficiency
2. **Sparse matrix synapses** for 50-100x propagation speedup
3. **6-layer D3M columns** for true generative capabilities
4. **Small-world topology** for biological connectivity
5. **Calcium-based STDP** for compartment-specific learning

**Timeline**: 12 weeks from design to full implementation  
**Risk Level**: Medium (proven components, significant refactor)  
**Reward**: 10-100x performance, neuromorphic-grade architecture

**Next Steps:**
1. Review and approve design
2. Create detailed Phase 1 implementation plan
3. Set up benchmarking infrastructure
4. Begin ALIF neuron implementation

---

**Document Version**: 1.0  
**Author**: Claude (with Kyle's specifications)  
**Status**: Ready for Review