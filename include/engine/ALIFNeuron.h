/**
 * @file ALIFNeuron.h
 * @brief Adaptive Leaky Integrate-and-Fire (ALIF) Neuron Model
 * 
 * This file defines the new ALIF neuron model for NeuroGen 2.0's cortical column
 * architecture. Key features:
 * - Structure of Arrays (SoA) layout for optimal GPU memory coalescing
 * - Simplified dynamics: 5 FLOPs per neuron per timestep (vs 40 for Izhikevich)
 * - 4x memory reduction: 32 bytes per neuron (vs 128 bytes)
 * - Preserved biological features: adaptation, refractory periods, dynamic thresholds
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef ALIF_NEURON_H
#define ALIF_NEURON_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath> // Added for expf

namespace neurogen {
namespace cortical {

// ============================================================================
// ALIF NEURON PARAMETERS
// ============================================================================

/**
 * @brief Default ALIF neuron parameters (biologically plausible)
 */
struct ALIFParameters {
    // Membrane dynamics
    float tau_mem = 20.0f;              // Membrane time constant (ms)
    float tau_adaptation = 100.0f;      // Adaptation time constant (ms)
    float v_rest = -70.0f;              // Resting potential (mV)
    float v_reset = -70.0f;             // Reset potential after spike (mV)
    float v_threshold_base = -50.0f;    // Base firing threshold (mV)

    // Adaptation parameters
    float adaptation_increment = 0.1f;  // Threshold increase per spike
    float adaptation_decay = 0.95f;     // Per-timestep decay factor

    // Refractory period
    float refractory_period = 2.0f;     // Absolute refractory period (ms)

    // Input scaling
    float input_resistance = 100.0f;    // Input resistance (MΩ)

    // Noise (for stochastic spiking)
    float noise_sigma = 0.0f;           // Membrane noise std dev

    // === SPIKE AMPLITUDE PARAMETERS (Weighted Spike Output) ===
    // These parameters control the float-valued spike output, enabling
    // neurons to transmit weighted signals while maintaining sparse gating.

    // Spike amplitude mode:
    // 0 = Fixed amplitude (use base_spike_amplitude)
    // 1 = Threshold-relative (amplitude = (V - theta) / scale + base)
    // 2 = Membrane-proportional (amplitude = (V - v_rest) / (theta - v_rest))
    // 3 = Learnable per-neuron amplitude (uses d_spike_amplitude array)
    int spike_amplitude_mode = 1;

    float base_spike_amplitude = 1.0f;  // Base/fixed spike amplitude
    float amplitude_scale = 20.0f;      // Scale factor for threshold-relative mode
    float min_spike_amplitude = 0.1f;   // Minimum spike amplitude (prevents vanishing)
    float max_spike_amplitude = 3.0f;   // Maximum spike amplitude (prevents explosion)

    // Compute decay factors from time constants
    __host__ __device__ float alpha_mem(float dt) const {
        return expf(-dt / tau_mem);
    }

    __host__ __device__ float alpha_adapt(float dt) const {
        return expf(-dt / tau_adaptation);
    }
};

// ============================================================================
// ALIF NEURON STATE - Structure of Arrays (SoA) Layout
// ============================================================================

/**
 * @brief ALIF Neuron State using Structure of Arrays (SoA) for GPU efficiency
 *
 * Memory Layout Comparison:
 * - Old (AoS): neurons[i].voltage requires loading entire struct (128 bytes)
 * - New (SoA): voltage[i] loads only 4 bytes, perfect coalescing
 *
 * Each array is allocated separately on GPU for maximum bandwidth utilization.
 *
 * === WEIGHTED SPIKE OUTPUT MODEL ===
 * This implementation uses float-valued spike outputs with gating:
 * - d_spike_output: Float value representing weighted spike amplitude
 * - Gating is implicit: spike_output = 0.0f means no spike (gate closed)
 * - Non-zero values carry both the "spike event" and its "intensity"
 *
 * This enables:
 * - Higher information capacity per spike (vs binary 0/1)
 * - Mimics biological synaptic facilitation and neural bursts
 * - GPU-efficient (native float operations on CUDA cores)
 * - Easier backpropagation (differentiable amplitude)
 */
struct ALIFNeuronArrays {
    // === PRIMARY STATE VARIABLES (32 bytes per neuron total) ===
    float* d_voltage;           // Membrane potential V[i] (mV)
    float* d_adaptation;        // Adaptive threshold A[i]
    float* d_threshold;         // Current threshold θ[i] = θ_base + A[i]
    float* d_current;           // Input current I[i] (nA)

    // === TIMING ===
    float* d_last_spike_time;   // Time of last spike (ms)
    float* d_refractory_remaining; // Remaining refractory time (ms)

    // === SPIKE OUTPUT (Weighted/Gated) ===
    float* d_spike_output;      // Weighted spike amplitude (0.0 = no spike, >0 = spike with amplitude)
    uint32_t* d_spike_count;    // Cumulative spike count (for monitoring)
    float* d_spike_amplitude;   // Learnable per-neuron amplitude factor (mode 3)

    // === ACTIVITY TRACKING ===
    float* d_firing_rate;       // Exponential moving average firing rate
    float* d_activity_trace;    // Activity trace for STDP

    // === NEURON METADATA (read-only after init) ===
    int8_t* d_neuron_type;      // 0=excitatory, 1=inhibitory, 2=modulatory
    int16_t* d_layer_id;        // Cortical layer (0-5 for L1-L6)
    int16_t* d_column_id;       // Cortical column index

    // Number of neurons
    int num_neurons;

    // Host-side parameters (constant memory on GPU)
    ALIFParameters params;
    
    // === MEMORY MANAGEMENT ===
    
    /**
     * @brief Allocate GPU memory for all arrays
     */
    cudaError_t allocate(int n_neurons) {
        num_neurons = n_neurons;
        cudaError_t err;

        // Primary state (32 bytes per neuron)
        err = cudaMalloc(&d_voltage, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_adaptation, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_threshold, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_current, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;

        // Timing
        err = cudaMalloc(&d_last_spike_time, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_refractory_remaining, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;

        // Spike output (float for weighted spikes)
        err = cudaMalloc(&d_spike_output, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_spike_count, n_neurons * sizeof(uint32_t));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_spike_amplitude, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;

        // Activity tracking
        err = cudaMalloc(&d_firing_rate, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_activity_trace, n_neurons * sizeof(float));
        if (err != cudaSuccess) return err;

        // Metadata
        err = cudaMalloc(&d_neuron_type, n_neurons * sizeof(int8_t));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_layer_id, n_neurons * sizeof(int16_t));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_column_id, n_neurons * sizeof(int16_t));
        if (err != cudaSuccess) return err;

        return cudaSuccess;
    }
    
    /**
     * @brief Free all GPU memory
     */
    void free() {
        if (d_voltage) cudaFree(d_voltage);
        if (d_adaptation) cudaFree(d_adaptation);
        if (d_threshold) cudaFree(d_threshold);
        if (d_current) cudaFree(d_current);
        if (d_last_spike_time) cudaFree(d_last_spike_time);
        if (d_refractory_remaining) cudaFree(d_refractory_remaining);
        if (d_spike_output) cudaFree(d_spike_output);
        if (d_spike_count) cudaFree(d_spike_count);
        if (d_spike_amplitude) cudaFree(d_spike_amplitude);
        if (d_firing_rate) cudaFree(d_firing_rate);
        if (d_activity_trace) cudaFree(d_activity_trace);
        if (d_neuron_type) cudaFree(d_neuron_type);
        if (d_layer_id) cudaFree(d_layer_id);
        if (d_column_id) cudaFree(d_column_id);

        d_voltage = d_adaptation = d_threshold = d_current = nullptr;
        d_last_spike_time = d_refractory_remaining = nullptr;
        d_spike_output = d_spike_amplitude = nullptr;
        d_spike_count = nullptr;
        d_firing_rate = d_activity_trace = nullptr;
        d_neuron_type = nullptr;
        d_layer_id = d_column_id = nullptr;
        num_neurons = 0;
    }
    
    /**
     * @brief Initialize all neurons to resting state
     */
    cudaError_t initialize();
    
    /**
     * @brief Get memory footprint in bytes
     */
    size_t getMemoryFootprint() const {
        // 4 floats primary state (16) + 2 timing (8) + 3 spike output (12: output + count + amplitude)
        // + 2 activity (8) + 1 byte type + 2 bytes layer + 2 bytes column
        return num_neurons * (4*4 + 2*4 + 4 + 4 + 4 + 2*4 + 1 + 2 + 2);
        // = num_neurons * 49 bytes (vs ~128 bytes in old model)
        // Slightly larger than before (was 44) due to float spike_output and spike_amplitude
    }
};

// ============================================================================
// MULTI-COMPARTMENT ALIF (for cortical columns requiring dendritic computation)
// ============================================================================

/**
 * @brief Compartmental ALIF for dendritic processing
 *
 * Used for pyramidal cells in cortical columns that need:
 * - Basal dendrite input (feedforward)
 * - Apical dendrite input (feedback/context)
 * - Somatic integration
 *
 * === WEIGHTED SPIKE OUTPUT ===
 * Like ALIFNeuronArrays, uses float-valued spike outputs with gating.
 */
struct CompartmentalALIFArrays {
    // Compartment count per neuron (typically 3: soma, basal, apical)
    static constexpr int NUM_COMPARTMENTS = 3;
    enum Compartment { SOMA = 0, BASAL = 1, APICAL = 2 };

    // === COMPARTMENT VOLTAGES ===
    float* d_v_soma;            // Somatic voltage
    float* d_v_basal;           // Basal dendrite voltage
    float* d_v_apical;          // Apical dendrite voltage

    // === COMPARTMENT CURRENTS ===
    float* d_i_soma;            // Current to soma
    float* d_i_basal;           // Current to basal dendrite
    float* d_i_apical;          // Current to apical dendrite

    // === CALCIUM (per compartment for STDP) ===
    float* d_ca_soma;           // Somatic calcium
    float* d_ca_basal;          // Basal calcium
    float* d_ca_apical;         // Apical calcium

    // === COUPLING CONDUCTANCES ===
    float* d_g_basal_soma;      // Basal-to-soma coupling
    float* d_g_apical_soma;     // Apical-to-soma coupling

    // === INHERITED FROM BASIC ALIF ===
    float* d_adaptation;
    float* d_threshold;
    float* d_last_spike_time;
    float* d_refractory_remaining;
    float* d_spike_output;      // Weighted spike amplitude (float, not binary)
    uint32_t* d_spike_count;
    float* d_spike_amplitude;   // Learnable per-neuron amplitude factor
    float* d_firing_rate;
    float* d_activity_trace;
    int8_t* d_neuron_type;
    int16_t* d_layer_id;
    int16_t* d_column_id;

    int num_neurons;
    ALIFParameters params;

    // Compartmental coupling parameters
    float g_coupling_basal = 0.5f;    // Basal-soma conductance
    float g_coupling_apical = 0.3f;   // Apical-soma conductance (weaker for context)

    cudaError_t allocate(int n_neurons);
    void free();
    cudaError_t initialize();

    size_t getMemoryFootprint() const {
        // 3 voltages + 3 currents + 3 calcium + 2 coupling + base ALIF (now 49 bytes)
        return num_neurons * (3*4 + 3*4 + 3*4 + 2*4 + 49);
        // = num_neurons * 93 bytes
    }
};

// ============================================================================
// NEURON TYPE DEFINITIONS
// ============================================================================

/**
 * @brief Neuron types for cortical columns
 */
enum class NeuronType : int8_t {
    // Excitatory neurons
    PYRAMIDAL = 0,              // Regular spiking pyramidal
    SPINY_STELLATE = 1,         // L4 spiny stellate cells
    
    // Inhibitory interneurons
    PV_BASKET = 10,             // Fast-spiking parvalbumin basket cells
    PV_CHANDELIER = 11,         // Parvalbumin chandelier cells
    SOM_MARTINOTTI = 12,        // Somatostatin Martinotti cells
    VIP = 13,                   // VIP interneurons (disinhibition)
    
    // Modulatory
    CHOLINERGIC = 20,           // Cholinergic interneurons
};

/**
 * @brief Get ALIF parameters for specific neuron types
 */
inline ALIFParameters getParametersForType(NeuronType type) {
    ALIFParameters p;
    
    switch (type) {
        case NeuronType::PYRAMIDAL:
            p.tau_mem = 20.0f;
            p.tau_adaptation = 100.0f;
            p.adaptation_increment = 0.1f;
            p.v_threshold_base = -50.0f;
            break;
            
        case NeuronType::PV_BASKET:
            // Fast-spiking: short time constants, minimal adaptation
            p.tau_mem = 10.0f;
            p.tau_adaptation = 20.0f;
            p.adaptation_increment = 0.02f;  // Minimal adaptation
            p.v_threshold_base = -45.0f;     // Lower threshold
            p.refractory_period = 1.0f;      // Short refractory
            break;
            
        case NeuronType::SOM_MARTINOTTI:
            // Low-threshold spiking: slower, more adaptation
            p.tau_mem = 30.0f;
            p.tau_adaptation = 200.0f;
            p.adaptation_increment = 0.2f;   // Strong adaptation
            p.v_threshold_base = -55.0f;
            break;
            
        default:
            // Use defaults
            break;
    }
    
    return p;
}

} // namespace cortical
} // namespace neurogen

#endif // ALIF_NEURON_H
