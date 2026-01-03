#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace neurogen {

/**
 * @brief Structure of Arrays (SoA) layout for high-performance neuron state.
 * 
 * Replaces the previous Array of Structures (AoS) GPUNeuronState.
 * Optimized for coalesced memory access on GPU.
 */
struct TensorNetwork {
    // === Core Neuron State (LIF-A Model) ===
    float* d_voltage;       // [N] Membrane potential (v)
    float* d_adaptation;    // [N] Adaptation variable (a)
    float* d_threshold;     // [N] Dynamic threshold (thresh)
    
    // === Activity Tracking ===
    uint8_t* d_spikes;      // [N] Spike bitmask (current step)
    float* d_input_current; // [N] Accumulated synaptic inputs
    
    // === Refractory Period ===
    float* d_last_spike_time; // [N] Timestamp of last spike
    
    // === Metadata ===
    size_t num_neurons;
    size_t allocated_capacity;

    // === Device pointers ownership ===
    // This struct acts as a handle. Memory is managed by NeuralEngine.
};

} // namespace neurogen

