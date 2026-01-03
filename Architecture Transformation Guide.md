# NeuroGen 2.0 Architecture Transformation - Code Comparison Guide

This document provides detailed before/after code comparisons for the major architectural changes in the cortical column overhaul.

---

## Transformation 1: Neuron Model (Izhikevich → ALIF)

### BEFORE: Izhikevich Neuron with Multi-Compartments

**Data Structure (AoS - Bad for GPU):**
```cpp
// Current: Structure of Arrays (complex, cache-unfriendly)
struct GPUNeuronState {
    float voltage;                    // Membrane potential
    float recovery;                   // Recovery variable (u)
    float ca_conc[MAX_COMPARTMENTS];  // Calcium per compartment
    float last_spike_time;
    float spike_threshold;
    float I_syn_0, I_syn_1, I_syn_2, I_syn_3; // Per-compartment currents
    // Total: ~128 bytes per neuron
};

// Array of structures (cache misses)
GPUNeuronState* neurons = new GPUNeuronState[num_neurons];
```

**Update Kernel (Complex, Multiple Sub-Steps):**
```cpp
// From current FusedKernels.cu - Izhikevich integration
__global__ void izhikevich_update_kernel(
    GPUNeuronState* neurons,
    float dt,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& n = neurons[idx];
    
    // Izhikevich parameters (biologically detailed but computationally expensive)
    float a = 0.02f;
    float b = 0.2f;
    float c = -65.0f;
    float d = 8.0f;
    
    // Requires sub-stepping for numerical stability
    float sub_dt = dt / 4.0f;
    for (int step = 0; step < 4; step++) {
        // Sum compartmental currents
        float I_total = n.I_syn_0 + n.I_syn_1 + n.I_syn_2 + n.I_syn_3;
        
        // Izhikevich dynamics (2 multiplies, 3 adds per sub-step)
        float dv = (0.04f * n.voltage * n.voltage + 5.0f * n.voltage + 140.0f 
                    - n.recovery + I_total) * sub_dt;
        float du = a * (b * n.voltage - n.recovery) * sub_dt;
        
        n.voltage += dv;
        n.recovery += du;
        
        // Spike detection and reset
        if (n.voltage >= 30.0f) {
            n.voltage = c;
            n.recovery += d;
            // ... calcium updates, etc.
        }
    }
    
    // Total: ~40 FLOPs per neuron per timestep
}
```

**Memory Access Pattern (Cache Inefficient):**
```cpp
// Reading neuron 0, 1, 2... loads entire 128-byte structs
// Wastes bandwidth when only need voltage values
neurons[0].voltage; // Loads 128 bytes
neurons[1].voltage; // Loads another 128 bytes (no coalescing)
```

---

### AFTER: ALIF Neuron (Optimized)

**Data Structure (SoA - GPU-Friendly):**
```cpp
// NEW: Structure of Arrays (coalesced, cache-friendly)
struct ALIFNeuronArrays {
    // Separate contiguous arrays
    float* d_voltage;          // [num_neurons]
    float* d_adaptation;       // [num_neurons]
    float* d_threshold;        // [num_neurons]
    float* d_ca_conc;         // [num_neurons] (simplified to 1 compartment initially)
    uint8_t* d_spike_flags;   // [num_neurons] (binary, packed)
    float* d_last_spike_time; // [num_neurons]
    
    // Total: 32 bytes per neuron (4x reduction)
};

// Allocation (each array contiguous)
cudaMalloc(&arrays.d_voltage, num_neurons * sizeof(float));
cudaMalloc(&arrays.d_adaptation, num_neurons * sizeof(float));
// etc.
```

**Update Kernel (Simplified, Single Step):**
```cpp
// NEW: Adaptive LIF with discrete-time dynamics
__global__ void alif_update_kernel(
    float* __restrict__ voltage,          // Input/Output
    float* __restrict__ adaptation,       // Input/Output
    float* __restrict__ threshold,        // Input/Output
    const float* __restrict__ input_current, // Input
    uint8_t* __restrict__ spikes,        // Output
    float* __restrict__ last_spike_time, // Input/Output
    int num_neurons,
    float dt,
    float current_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // ALIF parameters (pre-computed for efficiency)
    const float alpha = 0.95f;      // exp(-dt/tau_m) for dt=1ms, tau_m=20ms
    const float beta = 0.90f;       // exp(-dt/tau_a) for tau_a=100ms
    const float delta = 1.5f;       // Adaptation increment
    const float V_rest = -65.0f;
    const float V_reset = -70.0f;
    const float Theta_base = -50.0f;
    
    // Load state (coalesced reads - all 256 threads read consecutive memory)
    float V = voltage[idx];
    float A = adaptation[idx];
    float Theta = threshold[idx];
    float I = input_current[idx];
    
    // === CORE DYNAMICS (1 FMA) ===
    // V_{t+1} = α*(V_t - V_rest) + V_rest + I_t - A_t
    V = alpha * (V - V_rest) + V_rest + I - A;
    
    // === SPIKE DETECTION ===
    bool spiked = (V > Theta);
    spikes[idx] = spiked ? 1 : 0;
    
    if (spiked) {
        // Spike occurred
        V = V_reset;
        A += delta;  // Increase adaptation
        Theta = Theta_base + 5.0f;  // Temporary refractory period
        last_spike_time[idx] = current_time;
    } else {
        // No spike: decay threshold back to baseline
        Theta = beta * Theta + (1.0f - beta) * Theta_base;
    }
    
    // Decay adaptation
    A *= beta;
    
    // Store state (coalesced writes)
    voltage[idx] = V;
    adaptation[idx] = A;
    threshold[idx] = Theta;
    
    // Total: ~5 FLOPs per neuron per timestep (8x reduction)
}
```

**Memory Access Pattern (Optimal):**
```cpp
// Thread 0, 1, 2... read consecutive floats (128-byte cache line)
voltage[0], voltage[1], voltage[2], voltage[3] // All in same cache line!
// Perfect coalescing: 32 threads read 128 bytes in single transaction
```

**Performance Comparison:**

| Metric | Izhikevich | ALIF | Improvement |
|--------|------------|------|-------------|
| FLOPs per neuron | 40 | 5 | 8x |
| Memory per neuron | 128 bytes | 32 bytes | 4x |
| Memory bandwidth | Random | Coalesced | 10-20x |
| Numerical stability | Requires sub-stepping | Single step | 4x |
| **Total Speedup** | - | - | **~30x** |

---

## Transformation 2: Synapse Model (Struct → Sparse Matrix)

### BEFORE: Individual Synapse Structs

**Data Structure:**
```cpp
// Current: Each synapse is an individual struct
struct GPUSynapse {
    int pre_neuron_idx;           // 4 bytes
    int post_neuron_idx;          // 4 bytes
    float weight;                 // 4 bytes
    float eligibility_trace;      // 4 bytes
    float learning_rate;          // 4 bytes
    float dopamine_sensitivity;   // 4 bytes
    int post_compartment;         // 4 bytes
    uint8_t active;              // 1 byte
    // Padding to alignment       // 3 bytes
    // Total: 32 bytes per synapse
};

// Array of millions of synapses
GPUSynapse* synapses = new GPUSynapse[num_synapses]; // 32 MB for 1M synapses
```

**Propagation Kernel (Inefficient):**
```cpp
// Current: Process each synapse individually (random memory access)
__global__ void propagate_synapses_kernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    int num_synapses,
    float current_time,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& syn = synapses[idx];
    
    if (!syn.active) return;
    
    // PROBLEM: Random memory access pattern
    int pre_idx = syn.pre_neuron_idx;   // Random location
    int post_idx = syn.post_neuron_idx; // Random location
    
    // Check if pre-synaptic neuron spiked
    bool pre_spiked = (fabsf(neurons[pre_idx].last_spike_time - current_time) < dt);
    
    if (pre_spiked) {
        // PROBLEM: Atomic operation per synapse (serialization)
        int compartment = syn.post_compartment;
        float weight = syn.weight;
        
        // Uncoalesced atomic writes (terrible performance)
        if (compartment == 0) {
            atomicAdd(&neurons[post_idx].I_syn_0, weight);
        } else if (compartment == 1) {
            atomicAdd(&neurons[post_idx].I_syn_1, weight);
        }
        // etc...
    }
    
    // PROBLEM: Each thread processes 1 synapse
    // With 1M synapses, need 1M/256 = 4000 thread blocks
    // Very poor GPU utilization
}
```

**Performance Issues:**
1. **Random Memory Access**: Pre/post neuron indices scattered in memory
2. **Atomic Contention**: Multiple synapses targeting same neuron
3. **Thread Divergence**: Active/inactive synapses cause different code paths
4. **Poor Parallelism**: Can't use GPU tensor cores

---

### AFTER: Block Sparse Row (BSR) Matrix

**Data Structure:**
```cpp
// NEW: Sparse matrix representation (block-optimized for GPUs)
struct SparseWeightMatrix {
    // BSR format: groups weights into dense 32×32 blocks
    float* d_weight_values;      // [num_blocks * 1024] - Dense blocks
    int* d_col_indices;          // [num_blocks] - Which block-column
    int* d_row_ptr;              // [num_neuron_blocks + 1] - Row start indices
    
    // Plasticity tracking (parallel to weights)
    float* d_eligibility_trace;  // [num_blocks * 1024]
    
    int num_blocks;
    int block_size;              // 32 (warp size)
    int num_neurons;
    
    // Example: For 100K neurons with 5% connectivity
    // num_blocks ≈ (100K/32) * (100K/32) * 0.05 ≈ 15K blocks
    // Memory: 15K * 1024 * 4 bytes = 60 MB (vs 200 MB for individual synapses)
};
```

**Propagation Kernel (Optimized SpMV):**
```cpp
// NEW: Sparse Matrix-Vector Multiply (leverages GPU architecture)
__global__ void sparse_spmv_propagation_kernel(
    const float* __restrict__ weight_values,   // [num_blocks * 1024]
    const int* __restrict__ col_indices,       // [num_blocks]
    const int* __restrict__ row_ptr,           // [num_neuron_blocks + 1]
    const uint8_t* __restrict__ pre_spikes,    // [num_neurons] - binary vector
    float* __restrict__ post_currents,         // [num_neurons] - output
    int num_neuron_blocks,
    int block_size
) {
    // Each thread block processes one row-block (32 post-synaptic neurons)
    int row_block_idx = blockIdx.x;
    if (row_block_idx >= num_neuron_blocks) return;
    
    int tid = threadIdx.x; // 0-31 (one warp)
    int post_neuron = row_block_idx * block_size + tid;
    
    // Shared memory for weight block (32×32 matrix)
    __shared__ float weights[32][33]; // +1 to avoid bank conflicts
    
    // Accumulated current for this post-neuron
    float accumulated_current = 0.0f;
    
    // Iterate over non-zero blocks in this row
    int block_start = row_ptr[row_block_idx];
    int block_end = row_ptr[row_block_idx + 1];
    
    for (int b = block_start; b < block_end; ++b) {
        int col_block_idx = col_indices[b];
        int pre_neuron_base = col_block_idx * block_size;
        
        // === LOAD WEIGHT BLOCK INTO SHARED MEMORY ===
        // All 32 threads cooperate to load 32×32 = 1024 weights
        // Each thread loads 32 consecutive weights (coalesced!)
        for (int row = tid; row < block_size; row += 32) {
            for (int col = 0; col < block_size; ++col) {
                int weight_idx = b * 1024 + row * block_size + col;
                weights[row][col] = weight_values[weight_idx];
            }
        }
        __syncthreads();
        
        // === DOT PRODUCT WITH PRE-SYNAPTIC SPIKE VECTOR ===
        // Each thread computes: sum(weights[tid][j] * pre_spikes[pre_base + j])
        for (int j = 0; j < block_size; ++j) {
            int pre_idx = pre_neuron_base + j;
            if (pre_spikes[pre_idx]) {
                accumulated_current += weights[tid][j];
            }
        }
        __syncthreads();
    }
    
    // Write result (coalesced, no atomics needed!)
    post_currents[post_neuron] = accumulated_current;
    
    // KEY ADVANTAGES:
    // 1. Coalesced memory access (32 threads read consecutive memory)
    // 2. Shared memory reduces global memory traffic
    // 3. No atomic operations (each post-neuron handled by single thread)
    // 4. Can use tensor cores for FP16 (future optimization)
}
```

**Performance Comparison:**

| Metric | Individual Synapses | Sparse Matrix | Improvement |
|--------|---------------------|---------------|-------------|
| Memory per synapse | 32 bytes | ~4 bytes (amortized) | 8x |
| Memory access | Random | Coalesced | 20x |
| Atomic operations | Per synapse | None | Infinite |
| Tensor core usage | No | Yes (FP16) | 10x (future) |
| **Total Speedup** | - | - | **50-100x** |

**Example: 100K neurons, 1M synapses**
- **Old approach**: 1M kernel invocations, 1M atomics, 200 MB memory
- **New approach**: 3K kernel invocations, 0 atomics, 60 MB memory
- **Result**: ~80x faster propagation

---

## Transformation 3: Module → Cortical Column

### BEFORE: Brain Region Modules

**Module Design:**
```cpp
// Current: Each brain region is a separate entity
class CorticalModule {
    std::string module_name;  // "Wernicke", "Broca", etc.
    int num_neurons;          // Variable per module
    
    GPUNeuronState* d_neurons;
    GPUSynapse* d_synapses;
    
    // Module-specific parameters
    float dopamine_sensitivity;
    float inhibition_level;
    
    void update(float dt, float reward);
    void receiveInput(const std::vector<float>& input);
    std::vector<float> getOutputState();
};

// Brain Orchestrator manages distinct modules
class BrainOrchestrator {
    std::unique_ptr<CorticalModule> thalamus_;     // 30K neurons
    std::unique_ptr<CorticalModule> wernicke_;     // 300K neurons
    std::unique_ptr<CorticalModule> broca_;        // 300K neurons
    std::unique_ptr<CorticalModule> hippocampus_;  // 150K neurons
    std::unique_ptr<CorticalModule> pfc_;          // 300K neurons
    std::unique_ptr<CorticalModule> basal_ganglia_; // 75K neurons
    
    // Hardcoded connections
    std::vector<InterModuleConnection> connections_;
};
```

**Problems:**
1. **Not Scalable**: Adding new regions requires code changes
2. **Hardcoded Topology**: Connections manually defined
3. **No Layered Structure**: Missing cortical layer functionality
4. **Single Operating Mode**: Can't switch between perception/generation

---

### AFTER: Uniform Cortical Columns

**Column Design (6-Layer D3M Architecture):**
```cpp
// NEW: All modules are uniform cortical columns
class CorticalColumn {
    int column_id;
    int total_neurons;  // e.g., 5000
    
    // === 6-LAYER STRUCTURE ===
    struct LayerStructure {
        // Indices into neuron arrays (SoA layout)
        struct LayerRange {
            int start_idx;
            int end_idx;
            int num_neurons;
        };
        
        LayerRange layer1;   // 5% - Context Integration (apical input)
        LayerRange layer2_3; // 35% - Associative Memory (Key-Value store)
        LayerRange layer4;   // 20% - Sensory Input (Prediction Error)
        LayerRange layer5;   // 25% - Generative Output (Query/Burst)
        LayerRange layer6;   // 15% - Feedback Control (Attention)
    } layers;
    
    // === INTRA-COLUMN CONNECTIVITY (Sparse Matrices) ===
    // Bottom-up pathway (perception)
    SparseWeightMatrix L4_to_L23;      // Sensory → Association
    SparseWeightMatrix L23_recurrent;  // Lateral recurrence (memory)
    SparseWeightMatrix L23_to_L5;      // Association → Output
    
    // Top-down pathway (generation) - D3M!
    SparseWeightMatrix L5_to_L23;      // Query → Memory retrieval
    SparseWeightMatrix L6_to_L4;       // Prediction → Error calculation
    
    // Feedback pathways
    SparseWeightMatrix L1_to_L5_apical;  // Global context → Output modulation
    SparseWeightMatrix L6_to_TRN;        // Attention gating
    
    // === NEURON STATE (Shared SoA arrays) ===
    ALIFNeuronArrays neuron_state;
    
    // === OPERATING MODE ===
    enum class Mode {
        PERCEPTION,  // Bottom-up: L4 → L2/3 → L5
        GENERATION,  // Top-down: L5 → L2/3 → L6
        ATTENTION    // Selective gating
    };
    Mode current_mode;
    
    // === COLUMN TYPE (Determines learning rule) ===
    enum class ColumnType {
        SENSORY,      // Pure STDP (like old Thalamus/Wernicke)
        ASSOCIATIVE,  // Hybrid learning (like old Hippocampus/PFC)
        MOTOR         // Reward-modulated (like old Broca/Basal Ganglia)
    };
    ColumnType type;
    
    // === UPDATE METHODS ===
    void updatePerceptionMode(float dt);
    void updateGenerationMode(float dt);
    void switchMode(Mode new_mode);
};
```

**Network Design (Small-World Topology):**
```cpp
// NEW: Network of uniform columns with emergent specialization
class ColumnNetwork {
    // === ALL COLUMNS ARE IDENTICAL STRUCTURE ===
    std::vector<CorticalColumn> columns;  // e.g., 100-200 columns
    
    // === NETWORK TOPOLOGY ===
    struct Topology {
        // Macro-columns (clusters of related columns)
        struct MacroColumn {
            std::vector<int> column_indices;
            std::string semantic_role;  // "sensory", "semantic", "motor"
        };
        std::vector<MacroColumn> macro_columns;
        
        // Inter-column connectivity (sparse matrix)
        SparseWeightMatrix long_range_connections;
        
        // Rich club hubs (high-degree columns)
        std::vector<int> hub_column_indices;
        
        // Small-world metrics
        float clustering_coefficient;
        float average_path_length;
    };
    Topology topology;
    
    // === GLOBAL WORKSPACE (Thalamic Broadcasting) ===
    struct GlobalWorkspace {
        int active_column;  // "Winner" column broadcasting
        float* d_broadcast_state;  // Sent to all Layer 1s
        
        void broadcast(int winner_column_id);
        void updateAttention();
    };
    GlobalWorkspace workspace;
    
    // === UPDATE PIPELINE ===
    void update(float dt) {
        // 1. Sensory columns process input (PERCEPTION mode)
        for (auto& col : getSensoryColumns()) {
            col.updatePerceptionMode(dt);
        }
        
        // 2. Association columns integrate (hybrid mode)
        for (auto& col : getAssociativeColumns()) {
            col.updatePerceptionMode(dt);
        }
        
        // 3. Competition: determine "winning" column
        int winner = detectWinnerColumn();
        
        // 4. Global broadcast to all columns
        workspace.broadcast(winner);
        
        // 5. Motor columns generate output (GENERATION mode)
        for (auto& col : getMotorColumns()) {
            col.updateGenerationMode(dt);
        }
    }
    
    // === MIGRATION FROM OLD ARCHITECTURE ===
    void loadLegacyModule(const std::string& module_name, 
                          const std::vector<float>& old_weights) {
        // Map old "Broca" → Motor column cluster
        // Map old "Wernicke" → Semantic column cluster
        // etc.
    }
};
```

**Comparison:**

| Aspect | Old (Module-Based) | New (Column-Based) | Benefit |
|--------|-------------------|-------------------|---------|
| Scalability | Manual coding | Add columns dynamically | Infinite |
| Topology | Hardcoded | Small-world algorithm | Biological |
| Layering | None | 6-layer D3M | Generative |
| Uniformity | Custom per module | All columns identical | Simplicity |
| Specialization | Hardcoded | Emergent via learning | Flexibility |

---

## Transformation 4: Learning Rule Enhancement

### BEFORE: Simple STDP + Global Dopamine

**Current Implementation:**
```cpp
// From EnhancedSTDPKernel.cu
__global__ void enhanced_stdp_kernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& syn = synapses[idx];
    
    // Simple timing difference
    float delta_t = neurons[syn.post_neuron_idx].last_spike_time - 
                    neurons[syn.pre_neuron_idx].last_spike_time;
    
    // Basic STDP curve
    float plasticity = 0.0f;
    if (delta_t > 0 && delta_t < 20.0f) {
        plasticity = __expf(-delta_t / 10.0f) * 0.01f;  // LTP
    } else if (delta_t < 0 && delta_t > -20.0f) {
        plasticity = -__expf(delta_t / 10.0f) * 0.008f;  // LTD
    }
    
    // PROBLEM: Uses single global calcium value
    float ca_factor = neurons[syn.post_neuron_idx].ca_conc[0];
    plasticity *= ca_factor;
    
    // PROBLEM: Global dopamine broadcast (not compartment-specific)
    float dopamine = 1.0f;  // Passed globally
    plasticity *= dopamine * syn.dopamine_sensitivity;
    
    // Update weight
    syn.weight += plasticity * dt;
}
```

**Problems:**
1. No compartment-specific calcium dynamics
2. Global dopamine (can't have region-specific learning)
3. No eligibility trace decay
4. No layer-specific learning rates

---

### AFTER: Multi-Factor Compartmental STDP

**New Implementation:**
```cpp
// NEW: Calcium-gated, compartment-specific, layer-aware STDP
__global__ void advanced_stdp_kernel(
    // Sparse matrix representation
    float* __restrict__ weights,              // [num_blocks * 1024]
    float* __restrict__ eligibility_traces,   // [num_blocks * 1024]
    float* __restrict__ calcium_conc,         // [num_neurons * num_compartments]
    
    // Network state
    const uint8_t* __restrict__ pre_spikes,
    const uint8_t* __restrict__ post_spikes,
    const float* __restrict__ pre_spike_times,
    const float* __restrict__ post_spike_times,
    
    // Sparse matrix structure
    const int* __restrict__ col_indices,
    const int* __restrict__ row_ptr,
    
    // Learning parameters (per layer)
    const float* __restrict__ layer_learning_rates,  // [6] - one per layer
    const float* __restrict__ layer_ca_thresholds,   // [6]
    const int* __restrict__ neuron_layer_map,        // [num_neurons] → layer ID
    
    // Neuromodulation (spatially distributed)
    const float* __restrict__ dopamine_field,  // [num_columns] - per-column dopamine
    const int* __restrict__ neuron_column_map, // [num_neurons] → column ID
    
    int num_blocks,
    int block_size,
    float dt,
    float current_time
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;
    
    // Determine which row-block and column-block
    int row_block = findRowBlock(block_idx, row_ptr);
    int col_block = col_indices[block_idx];
    
    // Process each synapse in the 32×32 block
    for (int local_idx = 0; local_idx < block_size * block_size; ++local_idx) {
        int syn_idx = block_idx * 1024 + local_idx;
        
        int post_local = local_idx / block_size;
        int pre_local = local_idx % block_size;
        
        int post_neuron = row_block * block_size + post_local;
        int pre_neuron = col_block * block_size + pre_local;
        
        // === STEP 1: SPIKE TIMING ===
        float delta_t = post_spike_times[post_neuron] - pre_spike_times[pre_neuron];
        
        float stdp_magnitude = 0.0f;
        if (fabsf(delta_t) < 20.0f) {
            if (delta_t > 0) {
                stdp_magnitude = __expf(-delta_t / 10.0f);  // LTP
            } else {
                stdp_magnitude = -0.8f * __expf(delta_t / 10.0f);  // LTD
            }
        }
        
        // === STEP 2: CALCIUM GATING (Compartment-Specific) ===
        // NEW: Use post-synaptic neuron's compartment calcium
        int post_layer = neuron_layer_map[post_neuron];
        int compartment = getTargetCompartment(pre_neuron, post_neuron, post_layer);
        
        float ca = calcium_conc[post_neuron * 4 + compartment];
        float ca_threshold = layer_ca_thresholds[post_layer];
        
        // Calcium-dependent plasticity (supralinear for high Ca)
        float ca_factor = 1.0f;
        if (ca > ca_threshold * 1.5f) {
            ca_factor = 2.5f;  // Strong potentiation
        } else if (ca > ca_threshold) {
            ca_factor = 1.5f;  // Moderate potentiation
        } else if (ca < ca_threshold * 0.5f) {
            ca_factor = 0.5f;  // Weak plasticity
        }
        
        stdp_magnitude *= ca_factor;
        
        // === STEP 3: UPDATE ELIGIBILITY TRACE ===
        // Decay existing trace
        float tau_eligibility = 100.0f;  // 100ms decay
        eligibility_traces[syn_idx] *= __expf(-dt / tau_eligibility);
        
        // Add new STDP contribution
        eligibility_traces[syn_idx] += stdp_magnitude;
        
        // Clamp to prevent explosion
        eligibility_traces[syn_idx] = fmaxf(-2.0f, fminf(2.0f, eligibility_traces[syn_idx]));
        
        // === STEP 4: SPATIALLY-DISTRIBUTED DOPAMINE ===
        // NEW: Dopamine is per-column, not global
        int column_id = neuron_column_map[post_neuron];
        float dopamine = dopamine_field[column_id];
        
        // === STEP 5: LAYER-SPECIFIC LEARNING RATE ===
        float learning_rate = layer_learning_rates[post_layer];
        
        // === STEP 6: THREE-FACTOR WEIGHT UPDATE ===
        // ΔW = η * dopamine * eligibility * dt
        float weight_change = learning_rate * dopamine * eligibility_traces[syn_idx] * dt;
        
        // Update weight
        weights[syn_idx] += weight_change;
        
        // Clamp to valid range
        weights[syn_idx] = fmaxf(0.0f, fminf(2.0f, weights[syn_idx]));
    }
}

// Companion kernel: Update calcium based on spike coincidences
__global__ void update_calcium_kernel(
    float* __restrict__ calcium_conc,          // [num_neurons * 4]
    const uint8_t* __restrict__ pre_spikes,
    const uint8_t* __restrict__ post_spikes,
    const int* __restrict__ synapse_targets,   // Which compartment
    int num_neurons,
    float dt
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;
    
    const float tau_ca = 50.0f;  // 50ms calcium decay
    const float ca_increment = 0.5f;
    
    // Decay all compartments
    for (int comp = 0; comp < 4; ++comp) {
        int idx = neuron_idx * 4 + comp;
        calcium_conc[idx] *= __expf(-dt / tau_ca);
        
        // Increment on coincident activity
        // (Would need to check which synapses target this compartment - simplified here)
        bool spike_pair = post_spikes[neuron_idx];  // Simplified
        if (spike_pair) {
            calcium_conc[idx] += ca_increment;
        }
        
        // Saturation
        calcium_conc[idx] = fminf(calcium_conc[idx], 2.0f);
    }
}
```

**New Capabilities:**

| Feature | Old | New | Benefit |
|---------|-----|-----|---------|
| Calcium dynamics | Global | Per-compartment | Dendritic computation |
| Dopamine | Broadcast | Per-column | Regional learning |
| Learning rate | Fixed | Per-layer | Layer specialization |
| Eligibility trace | Static | Decaying | Proper credit assignment |
| STDP curve | Fixed | Calcium-gated | Nonlinear plasticity |

---

## Transformation 5: D3M Generative Loops

### BEFORE: Unidirectional Processing

**Current Flow (Feedforward Only):**
```cpp
// From BrainOrchestrator::cognitiveStep()

// PROBLEM: Only one direction
void BrainOrchestrator::cognitiveStep(const std::vector<float>& input) {
    // 1. Input → Thalamus
    thalamus_->receiveInput(input);
    thalamus_->update(dt, reward);
    
    // 2. Thalamus → Wernicke
    auto thalamic_out = thalamus_->getOutputState();
    wernicke_->receiveInput(thalamic_out);
    wernicke_->update(dt, reward);
    
    // 3. Wernicke → PFC
    auto semantic_out = wernicke_->getOutputState();
    pfc_->receiveInput(semantic_out);
    pfc_->update(dt, reward);
    
    // 4. PFC → Broca
    auto pfc_out = pfc_->getOutputState();
    broca_->receiveInput(pfc_out);
    broca_->update(dt, reward);
    
    // 5. Broca → Output
    output = broca_->getOutputState();
    
    // MISSING: No top-down generative pathway!
    // Cannot do memory recall, prediction, imagination
}
```

---

### AFTER: Bidirectional D3M Loops

**New Flow (Perception & Generation Modes):**
```cpp
// NEW: Bidirectional processing with mode switching

class CorticalColumn {
    // ... previous definitions ...
    
    // === PERCEPTION MODE (Bottom-Up) ===
    void updatePerceptionMode(float dt) {
        // L4 receives input from thalamus
        computeL4PredictionError();
        
        // L4 → L2/3 (forward propagation)
        propagateL4toL23();
        
        // L2/3 recurrent dynamics (lateral associations)
        updateL23Recurrence(dt);
        
        // L2/3 → L5 (context to output)
        propagateL23toL5();
        
        // L5 bursting (if basal + apical coincidence)
        detectL5Bursts();
    }
    
    // === GENERATION MODE (Top-Down) ===
    void updateGenerationMode(float dt) {
        // L5 generates query (from internal state or global broadcast)
        generateL5Query();
        
        // L5 → L2/3 BACKWARD (query memory)
        propagateL5toL23();  // D3M backward connection!
        
        // L2/3 retrieves associated pattern
        retrieveL23Pattern();
        
        // L2/3 → L6 (retrieved pattern to feedback)
        propagateL23toL6();
        
        // L6 → L4 (generate prediction)
        generateL6Prediction();
        
        // L4 compares prediction vs. input (prediction error)
        computeL4PredictionError();
    }
};

// Kernels for D3M backward propagation
__global__ void propagate_L5_to_L23_backward_kernel(
    const float* __restrict__ layer5_query,     // L5 neuron activity
    const float* __restrict__ weights_L5_to_L23,  // Backward weights (fixed or learned)
    float* __restrict__ layer23_retrieval,      // Output: retrieved pattern
    int num_layer5,
    int num_layer23
) {
    int l23_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (l23_idx >= num_layer23) return;
    
    float retrieval = 0.0f;
    
    // Dot product: L5_query · W^T
    for (int l5_idx = 0; l5_idx < num_layer5; ++l5_idx) {
        float weight = weights_L5_to_L23[l5_idx * num_layer23 + l23_idx];
        retrieval += layer5_query[l5_idx] * weight;
    }
    
    // Apply activation (e.g., ReLU)
    layer23_retrieval[l23_idx] = fmaxf(0.0f, retrieval);
}
```

**Mode Switching Logic:**
```cpp
void ColumnNetwork::update(float dt) {
    // Determine operating mode based on input presence
    bool has_external_input = (input_signal_strength > threshold);
    
    if (has_external_input) {
        // === PERCEPTION MODE ===
        for (auto& col : sensory_columns) {
            col.updatePerceptionMode(dt);
        }
        for (auto& col : association_columns) {
            col.updatePerceptionMode(dt);
        }
    } else {
        // === AUTONOMOUS GENERATION MODE ===
        // Columns operate in "thinking" mode
        for (auto& col : association_columns) {
            col.updateGenerationMode(dt);
        }
        
        // Global workspace broadcasts "thought" to all columns
        int active_column = detectMostActiveColumn();
        broadcastToLayer1(active_column);
    }
    
    // Motor columns always operate in generation mode
    for (auto& col : motor_columns) {
        col.updateGenerationMode(dt);
    }
}
```

**Comparison:**

| Capability | Old (Feedforward) | New (D3M Bidirectional) |
|------------|-------------------|-------------------------|
| Memory recall | No | Yes (L5 → L2/3) |
| Prediction | No | Yes (L6 → L4) |
| Autonomous thought | No | Yes (internal loops) |
| Active inference | No | Yes (prediction error minimization) |
| Generative modeling | No | Yes (top-down synthesis) |

---

## Summary: Overall Architecture Transformation

### Before (NeuroGen 1.x)
- **Neurons**: Izhikevich (complex, slow)
- **Synapses**: Individual structs (random access)
- **Modules**: Hardcoded brain regions
- **Learning**: Simple STDP + global dopamine
- **Flow**: Unidirectional (feedforward)
- **Scale**: ~100K neurons max

### After (NeuroGen 2.0-Cortical)
- **Neurons**: ALIF (simple, fast) - **8x speedup**
- **Synapses**: Sparse matrices (coalesced) - **50-100x speedup**
- **Modules**: Uniform cortical columns - **Infinitely scalable**
- **Learning**: Multi-factor compartmental STDP - **Biologically realistic**
- **Flow**: Bidirectional D3M loops - **Generative capabilities**
- **Scale**: 1M+ neurons achievable

### Expected Performance
- **Computational**: 10-30x faster neuron updates
- **Memory**: 4x reduction per neuron
- **Synaptic Propagation**: 50-100x faster
- **Overall**: ~100x throughput improvement
- **Scale**: 10x larger networks on same hardware

---

**Next Steps:**
1. Implement ALIF neuron kernel (Week 1)
2. Implement sparse matrix synapses (Week 2)
3. Build single cortical column (Week 3-4)
4. Create network topology (Week 5-6)
5. Migrate old modules (Week 7-8)

This transformation moves NeuroGen from a "brain-inspired" architecture to a true **neuromorphic** system capable of autonomous cognition.