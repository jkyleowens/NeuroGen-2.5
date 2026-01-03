NeuroGen 2.0: High-Performance Bio-Mimetic Architecture Refactor

Status: RFC (Request for Comments)
Target Version: 2.0.0
Objective: Maximizing parameter density and computational throughput via functional abstraction.

1. Executive Summary

The NeuroGen 1.x architecture successfully implements a high-fidelity biological simulation using Izhikevich neuron models and object-oriented synaptic structures. While scientifically accurate, this approach suffers from memory bandwidth saturation and thread divergence, limiting the maximum network size to ~100k neurons on consumer hardware.

NeuroGen 2.0 represents a paradigm shift from Simulation (replicating physics) to Emulation (replicating function). By abstracting neurons into Adaptive Linear Units (ALUs) and synapses into Compressed Sparse Row (CSR) matrices, we aim to increase the parameter count by 100x while reducing the computational footprint per update cycle.

2. Core Architectural Changes

2.1 The Neuron Abstraction (LIF-A)

We move from solving quadratic differential equations (Izhikevich) to discrete linear state updates. This preserves the essential spiking dynamics—refractory periods, adaptation, and threshold variability—without the overhead of floating-point intensive solvers.

Old Model (Izhikevich):

State: v (voltage), u (recovery), I (currents), Ca (calcium array)

Compute: 2 muls + 3 adds per sub-step (requires multiple sub-steps per global dt).

Memory: ~128 bytes per neuron (AoS).

New Model (LIF-A - Linear Integrate & Fire with Adaptation):

State: v (voltage), a (adaptation), thresh (dynamic threshold).

Compute: 1 fused multiply-add (FMA) per step.

Memory: ~16 bytes per neuron (SoA).

Dynamics:


$$V_{t+1} = \alpha V_t + \sum W_{in} - A_t$$

$$A_{t+1} = \beta A_t + (Spike ? \delta : 0)$$

2.2 The Synapse Abstraction (Sparse Matrices)

Current implementation treats synapses as individual C++ objects/structs. This creates "pointer chasing" and non-coalesced memory access.

New Approach: Block-Sparse Matrix Multiplication

Synapses are stored as weights in Block Compressed Sparse Row (BSR) format.

Update Step: The "Synaptic Input" kernel is replaced by a Sparse Matrix-Vector Multiplication (SpMV).

Input_Vector = All presynaptic spikes (0 or 1).

Weight_Matrix = All synaptic weights.

Current_Vector = Weight_Matrix * Input_Vector

Benefit: This utilizes the GPU's tensor cores and high-bandwidth memory (HBM) efficiently, as contiguous blocks of weights are read into cache simultaneously.

3. Structural Refactoring

3.1 Data Layout: Structure of Arrays (SoA)

The GPUNeuronState struct will be dissolved. The network state will be managed as parallel arrays in global memory.

Implementation:

class NeuralEngine {
    // Neuron States (Aligned Arrays)
    float* d_voltage;       // [N_neurons]
    float* d_adaptation;    // [N_neurons]
    float* d_threshold;     // [N_neurons]
    uint8_t* d_spikes;      // [N_neurons] (Bitmask)

    // Synaptic Weights (Sparse Matrix)
    float* d_weights_values; // [N_synapses]
    int* d_weights_indices;  // [N_synapses] (Column indices)
    int* d_weights_ptr;      // [N_neurons + 1] (Row pointers)
};


3.2 The "Hyper-Column" (Tensor Blocks)

The CorticalColumn concept will be mapped to CUDA Thread Blocks.

Old Way: Columns were logical groupings in C++.

New Way: A "Column" is a block of 256 or 512 neurons that fit exactly into GPU shared memory.

Local Inhibition: Within a Hyper-Column block, we enforce k-Winner-Take-All (kWTA) sparsity using fast intra-warp reduction (shuffles). This replaces complex inhibitory neuron simulations with a direct sorting/masking operation.

4. Learning Rule Refactor: Matrix-Plasticity

We replace the pairwise STDP check (checking every synapse against every spike) with a Three-Factor Matrix Rule.

Eligibility Trace (E):

When Pre-neuron fires: $E_{pre} += 1$ (and decays).

When Post-neuron fires: $E_{post} += 1$ (and decays).

The Update:

Instead of branching if (spike), we perform a masked matrix addition.

$\Delta W = \eta \cdot (S_{post}^T \times E_{pre} - E_{post}^T \times S_{pre})$

This allows plasticity updates to be batched as matrix operations.

5. Structural Plasticity & Pruning

To maintain the efficiency of the sparse architecture over time, we implement dynamic pruning strategies.

5.1 Synaptic Pruning (Continuous)

Criteria: During weight updates, if $|W_{ij}| < \epsilon$ (near-zero), the synapse is flagged.

Mechanism: When rebuilding the sparse matrix structure (BSR), blocks containing only pruned weights are removed entirely.

Benefit: Keeps the matrix sparse, ensuring that the "Sparse Matrix-Vector Multiplication" (SpMV) operation remains fast even as learning progresses.

5.2 Neuronal Pruning (Periodic "Sleep Cycle")

Removing neurons from a running GPU array is expensive (memory shifts). We use a periodic maintenance phase to handle this.

Trigger: Every $N$ steps (e.g., 10,000 steps), or during a designated "Sleep" phase.

Logic: 1. Scan spike_history for neurons with zero activity over the epoch.
2. Identify "Dead Nodes" (neurons with no incoming connections).
3. Perform Swap-Delete: Move the last active neuron in the array into the dead neuron's slot, update relevant indices, and decrement num_active_neurons.

Biological Analogy: Mimics the brain's metabolic cleanup and consolidation processes during sleep.

6. Migration Roadmap

Phase 1: Data Structures (Weeks 1-2)

Create engine/TensorNetwork.h to define the new SoA pointers.

Implement engine/SparseMatrix.cu to handle CSR/BSR format creation and access.

Milestone: Successfully allocate 1 million neurons and 1 billion synapses in VRAM.

Phase 2: The Compute Kernel (Weeks 3-4)

Implement kernels/LIF_Update.cu: A fused kernel that updates Voltage, Adaptation, and Spikes in one pass.

Implement kernels/SpMV_Input.cu: Using standard CuSparse or custom kernels to compute input currents.

Milestone: Validate that "resting state" behavior mimics NeuroGen 1.1 dynamics.

Phase 3: Plasticity & Training (Weeks 5-6)

Port the Reward Modulation logic to operate on the new global d_weights_values array.

Implement the Matrix-STDP update rule.

Implement Pruning logic (Sleep Cycle).

Milestone: Train on SlimPajama dataset and benchmark throughput (Tokens/Sec).

7. Expected Performance Gains

Metric

NeuroGen 1.1

NeuroGen 2.0 (Target)

Memory / Neuron

~128 bytes

~16 bytes

Max Neurons (16GB)

~200,000

~5,000,000

Update Speed

Limited by Memory Latency

Limited by Compute (Tensor Cores)

Parameter Count

~100 Million

~1 Billion+

8. Conclusion

This refactor abandons the comfortable familiarity of "Object-Oriented Neuroscience" for the raw performance of "Linear Algebra Neuroscience." It retains the spirit of the biological brain (spikes, sparsity, plasticity) but implements them in the native language of the GPU.