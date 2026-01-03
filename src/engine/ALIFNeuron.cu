/**
 * @file ALIFNeuron.cu
 * @brief Implementation of ALIF neuron array methods
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#include "engine/ALIFNeuron.h"
#include "engine/ALIFKernels.cuh"
#include <cuda_runtime.h>

namespace neurogen {
namespace cortical {

// ============================================================================
// ALIFNeuronArrays Implementation
// ============================================================================

cudaError_t ALIFNeuronArrays::initialize() {
    return kernels::launchALIFInit(*this);
}

// ============================================================================
// CompartmentalALIFArrays Implementation
// ============================================================================

cudaError_t CompartmentalALIFArrays::allocate(int n_neurons) {
    num_neurons = n_neurons;
    cudaError_t err;

    // Compartment voltages
    err = cudaMalloc(&d_v_soma, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_v_basal, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_v_apical, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;

    // Compartment currents
    err = cudaMalloc(&d_i_soma, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_i_basal, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_i_apical, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;

    // Calcium per compartment
    err = cudaMalloc(&d_ca_soma, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_ca_basal, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_ca_apical, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;

    // Coupling conductances
    err = cudaMalloc(&d_g_basal_soma, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_g_apical_soma, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;

    // Basic ALIF state
    err = cudaMalloc(&d_adaptation, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_threshold, n_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
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

void CompartmentalALIFArrays::free() {
    if (d_v_soma) cudaFree(d_v_soma);
    if (d_v_basal) cudaFree(d_v_basal);
    if (d_v_apical) cudaFree(d_v_apical);
    if (d_i_soma) cudaFree(d_i_soma);
    if (d_i_basal) cudaFree(d_i_basal);
    if (d_i_apical) cudaFree(d_i_apical);
    if (d_ca_soma) cudaFree(d_ca_soma);
    if (d_ca_basal) cudaFree(d_ca_basal);
    if (d_ca_apical) cudaFree(d_ca_apical);
    if (d_g_basal_soma) cudaFree(d_g_basal_soma);
    if (d_g_apical_soma) cudaFree(d_g_apical_soma);
    if (d_adaptation) cudaFree(d_adaptation);
    if (d_threshold) cudaFree(d_threshold);
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

    d_v_soma = d_v_basal = d_v_apical = nullptr;
    d_i_soma = d_i_basal = d_i_apical = nullptr;
    d_ca_soma = d_ca_basal = d_ca_apical = nullptr;
    d_g_basal_soma = d_g_apical_soma = nullptr;
    d_adaptation = d_threshold = nullptr;
    d_last_spike_time = d_refractory_remaining = nullptr;
    d_spike_output = d_spike_amplitude = nullptr;
    d_spike_count = nullptr;
    d_firing_rate = d_activity_trace = nullptr;
    d_neuron_type = nullptr;
    d_layer_id = d_column_id = nullptr;
    num_neurons = 0;
}

// Initialization kernel for compartmental neurons with weighted spike output
__global__ void compartmental_init_kernel(
    float* v_soma,
    float* v_basal,
    float* v_apical,
    float* i_soma,
    float* i_basal,
    float* i_apical,
    float* ca_soma,
    float* ca_basal,
    float* ca_apical,
    float* g_basal_soma,
    float* g_apical_soma,
    float* adaptation,
    float* threshold,
    float* last_spike_time,
    float* refractory_remaining,
    float* spike_output,
    uint32_t* spike_count,
    float* spike_amplitude,
    float* firing_rate,
    float* activity_trace,
    float v_rest,
    float v_threshold_base,
    float base_spike_amp,
    float g_basal_default,
    float g_apical_default,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    // Initialize all compartment voltages to resting
    v_soma[idx] = v_rest;
    v_basal[idx] = v_rest;
    v_apical[idx] = v_rest;

    // Zero currents
    i_soma[idx] = 0.0f;
    i_basal[idx] = 0.0f;
    i_apical[idx] = 0.0f;

    // Zero calcium
    ca_soma[idx] = 0.0f;
    ca_basal[idx] = 0.0f;
    ca_apical[idx] = 0.0f;

    // Default coupling conductances
    g_basal_soma[idx] = g_basal_default;
    g_apical_soma[idx] = g_apical_default;

    // ALIF state
    adaptation[idx] = 0.0f;
    threshold[idx] = v_threshold_base;
    last_spike_time[idx] = -1000.0f;
    refractory_remaining[idx] = 0.0f;
    spike_output[idx] = 0.0f;    // No spike (float weighted output)
    spike_count[idx] = 0;
    spike_amplitude[idx] = base_spike_amp;  // Initialize learnable amplitude
    firing_rate[idx] = 0.0f;
    activity_trace[idx] = 0.0f;
}

cudaError_t CompartmentalALIFArrays::initialize() {
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;

    compartmental_init_kernel<<<grid_size, block_size>>>(
        d_v_soma, d_v_basal, d_v_apical,
        d_i_soma, d_i_basal, d_i_apical,
        d_ca_soma, d_ca_basal, d_ca_apical,
        d_g_basal_soma, d_g_apical_soma,
        d_adaptation, d_threshold,
        d_last_spike_time, d_refractory_remaining,
        d_spike_output, d_spike_count, d_spike_amplitude,
        d_firing_rate, d_activity_trace,
        params.v_rest,
        params.v_threshold_base,
        params.base_spike_amplitude,
        g_coupling_basal,
        g_coupling_apical,
        num_neurons
    );

    return cudaGetLastError();
}

} // namespace cortical
} // namespace neurogen
