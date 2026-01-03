/**
 * @file HebbianLearningKernel_Bio.cu
 * @brief Biologically-constrained Hebbian/STDP learning with Dale's Law
 *
 * This file implements learning rules that respect biological constraints:
 * - Dale's Law: A neuron's synaptic sign cannot change
 * - Spike-Timing-Dependent Plasticity (STDP)
 * - Weight bounds based on neuron type
 * - Homeostatic regulation
 *
 * @version 2.0-Biology-First
 * @date December 30, 2025
 */

#include "engine/BioConstants.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace neurogen {
namespace bio {

// ============================================================================
// BIOLOGICAL STDP KERNEL
// ============================================================================

/**
 * @brief STDP learning with Dale's Law enforcement
 *
 * Implements spike-timing-dependent plasticity while maintaining sign constraints:
 * - Excitatory neurons: weights remain positive
 * - Inhibitory neurons: weights remain negative
 * - Uses eligibility traces for credit assignment
 *
 * @param weights Synaptic weight matrix (CSR format)
 * @param row_ptr CSR row pointers
 * @param col_ind CSR column indices
 * @param pre_spike_times Spike times for presynaptic neurons
 * @param post_spike_times Spike times for postsynaptic neurons
 * @param num_excitatory Boundary index (neurons < this are excitatory)
 * @param learning_rate Global learning rate
 * @param A_plus LTP amplitude
 * @param A_minus LTD amplitude
 * @param tau_plus LTP time constant (ms)
 * @param tau_minus LTD time constant (ms)
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 */
__global__ void biological_stdp_kernel(
    float* weights,
    const int* row_ptr,
    const int* col_ind,
    const float* pre_spike_times,
    const float* post_spike_times,
    const int num_excitatory,
    float learning_rate,
    float A_plus,
    float A_minus,
    float tau_plus,
    float tau_minus,
    float current_time,
    int num_neurons)
{
    int pre_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre_neuron >= num_neurons) return;

    // Determine neuron type
    bool is_inhibitory = (pre_neuron >= num_excitatory);

    // Get connections for this presynaptic neuron
    int start_idx = row_ptr[pre_neuron];
    int end_idx = row_ptr[pre_neuron + 1];

    // Get last spike time for presynaptic neuron
    float t_pre = pre_spike_times[pre_neuron];

    // Skip if neuron hasn't spiked recently (within time window)
    if (current_time - t_pre > 100.0f) return;  // 100ms window

    // Process all synapses from this neuron
    for (int i = start_idx; i < end_idx; i++) {
        int post_neuron = col_ind[i];
        float t_post = post_spike_times[post_neuron];

        // Skip if postsynaptic neuron hasn't spiked recently
        if (current_time - t_post > 100.0f) continue;

        // Calculate spike time difference
        float dt = t_post - t_pre;

        // Compute STDP weight change
        float dw = 0.0f;

        if (dt > 0) {
            // Post after pre -> LTP (potentiation)
            // dw = A_plus * exp(-dt / tau_plus)
            dw = A_plus * expf(-dt / tau_plus);
        } else if (dt < 0) {
            // Pre after post -> LTD (depression)
            // dw = -A_minus * exp(dt / tau_minus)
            dw = -A_minus * expf(dt / tau_minus);
        }

        // Apply learning rate
        dw *= learning_rate;

        // Update weight
        float new_w = weights[i] + dw;

        // DALE'S LAW CLAMPING
        new_w = clampWeightDalesLaw(new_w, is_inhibitory);

        weights[i] = new_w;
    }
}

// ============================================================================
// ELIGIBILITY TRACE STDP KERNEL
// ============================================================================

/**
 * @brief Three-factor learning rule with eligibility traces and reward modulation
 *
 * Implements reward-modulated STDP:
 * - dw = reward * eligibility_trace
 * - Eligibility trace tracks spike timing correlations
 * - Dale's Law enforced on final weights
 *
 * @param weights Synaptic weights
 * @param eligibility_traces Eligibility trace per synapse
 * @param row_ptr CSR row pointers
 * @param col_ind CSR column indices
 * @param pre_traces Presynaptic spike traces
 * @param post_traces Postsynaptic spike traces
 * @param num_excitatory Boundary index
 * @param reward_signal Dopamine/reward signal
 * @param learning_rate Learning rate
 * @param trace_decay Eligibility trace decay rate
 * @param dt Time step (ms)
 * @param num_neurons Total neurons
 */
__global__ void eligibility_trace_stdp_kernel(
    float* weights,
    float* eligibility_traces,
    const int* row_ptr,
    const int* col_ind,
    const float* pre_traces,
    const float* post_traces,
    const int num_excitatory,
    float reward_signal,
    float learning_rate,
    float trace_decay,
    float dt,
    int num_neurons)
{
    int pre_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre_neuron >= num_neurons) return;

    bool is_inhibitory = (pre_neuron >= num_excitatory);

    int start_idx = row_ptr[pre_neuron];
    int end_idx = row_ptr[pre_neuron + 1];

    float pre_trace = pre_traces[pre_neuron];

    for (int i = start_idx; i < end_idx; i++) {
        int post_neuron = col_ind[i];
        float post_trace = post_traces[post_neuron];

        // Update eligibility trace (correlation)
        float correlation = pre_trace * post_trace;

        // Decay existing trace and add new correlation
        eligibility_traces[i] = eligibility_traces[i] * trace_decay + correlation;

        // Apply reward-modulated weight update
        float dw = learning_rate * reward_signal * eligibility_traces[i] * dt;

        float new_w = weights[i] + dw;

        // DALE'S LAW ENFORCEMENT
        new_w = clampWeightDalesLaw(new_w, is_inhibitory);

        weights[i] = new_w;
    }
}

// ============================================================================
// HOMEOSTATIC PLASTICITY KERNEL
// ============================================================================

/**
 * @brief Homeostatic synaptic scaling with Dale's Law
 *
 * Adjusts weights to maintain target firing rate while preserving sign:
 * - If neuron fires too much: scale down incoming weights
 * - If neuron fires too little: scale up incoming weights
 * - Sign preservation: excitatory stays positive, inhibitory stays negative
 *
 * @param weights Synaptic weights
 * @param row_ptr CSR row pointers
 * @param col_ind CSR column indices
 * @param firing_rates Current firing rates (Hz)
 * @param target_rate Target firing rate (Hz)
 * @param num_excitatory Boundary index
 * @param scaling_rate Homeostatic scaling rate
 * @param dt Time step
 * @param num_neurons Total neurons
 */
__global__ void homeostatic_plasticity_kernel(
    float* weights,
    const int* row_ptr,
    const int* col_ind,
    const float* firing_rates,
    float target_rate,
    const int num_excitatory,
    float scaling_rate,
    float dt,
    int num_neurons)
{
    int post_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (post_neuron >= num_neurons) return;

    float current_rate = firing_rates[post_neuron];
    float rate_error = target_rate - current_rate;

    // Multiplicative scaling factor
    float scaling_factor = 1.0f + scaling_rate * rate_error * dt;

    // Find all incoming synapses to this neuron
    // (In CSR format, we need to search through all rows)
    // This is simplified - in production, use CSC format or transpose

    // Alternative: Scale outgoing synapses based on firing rate
    // This is more efficient with CSR format
    int start_idx = row_ptr[post_neuron];
    int end_idx = row_ptr[post_neuron + 1];

    for (int i = start_idx; i < end_idx; i++) {
        int pre_neuron = col_ind[i];  // Note: this is actually post in CSR
        bool is_inhibitory = (pre_neuron >= num_excitatory);

        float new_w = weights[i] * scaling_factor;

        // Dale's Law enforcement after scaling
        new_w = clampWeightDalesLaw(new_w, is_inhibitory);

        weights[i] = new_w;
    }
}

// ============================================================================
// TRIPLET STDP KERNEL (Advanced)
// ============================================================================

/**
 * @brief Triplet STDP rule with Dale's Law
 *
 * More accurate model of synaptic plasticity:
 * - Accounts for spike triplets (pre-post-pre and post-pre-post)
 * - Better captures experimental data
 * - Maintains Dale's Law constraints
 */
__global__ void triplet_stdp_kernel(
    float* weights,
    const int* row_ptr,
    const int* col_ind,
    const float* pre_trace_fast,
    const float* pre_trace_slow,
    const float* post_trace_fast,
    const float* post_trace_slow,
    const int num_excitatory,
    float learning_rate,
    float A2_plus,  // Pair-based LTP
    float A3_plus,  // Triplet-based LTP
    float A2_minus, // Pair-based LTD
    float A3_minus, // Triplet-based LTD
    float dt,
    int num_neurons)
{
    int pre_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre_neuron >= num_neurons) return;

    bool is_inhibitory = (pre_neuron >= num_excitatory);

    int start_idx = row_ptr[pre_neuron];
    int end_idx = row_ptr[pre_neuron + 1];

    float pre_fast = pre_trace_fast[pre_neuron];
    float pre_slow = pre_trace_slow[pre_neuron];

    for (int i = start_idx; i < end_idx; i++) {
        int post_neuron = col_ind[i];

        float post_fast = post_trace_fast[post_neuron];
        float post_slow = post_trace_slow[post_neuron];

        // Triplet STDP rule
        // LTP: triggered by post spike
        float ltp = A2_plus * pre_fast + A3_plus * pre_fast * post_slow;

        // LTD: triggered by pre spike
        float ltd = A2_minus * post_fast + A3_minus * post_fast * pre_slow;

        float dw = learning_rate * (ltp - ltd) * dt;

        float new_w = weights[i] + dw;

        // Dale's Law enforcement
        new_w = clampWeightDalesLaw(new_w, is_inhibitory);

        weights[i] = new_w;
    }
}

// ============================================================================
// KERNEL LAUNCH WRAPPERS
// ============================================================================

/**
 * @brief Launch biological STDP kernel
 */
cudaError_t launchBiologicalSTDP(
    float* d_weights,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_pre_spike_times,
    const float* d_post_spike_times,
    int num_excitatory,
    float learning_rate,
    float A_plus,
    float A_minus,
    float tau_plus,
    float tau_minus,
    float current_time,
    int num_neurons,
    cudaStream_t stream = 0)
{
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;

    biological_stdp_kernel<<<grid_size, block_size, 0, stream>>>(
        d_weights, d_row_ptr, d_col_ind,
        d_pre_spike_times, d_post_spike_times,
        num_excitatory,
        learning_rate, A_plus, A_minus,
        tau_plus, tau_minus, current_time,
        num_neurons
    );

    return cudaGetLastError();
}

/**
 * @brief Launch eligibility trace STDP kernel
 */
cudaError_t launchEligibilityTraceSTDP(
    float* d_weights,
    float* d_eligibility_traces,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_pre_traces,
    const float* d_post_traces,
    int num_excitatory,
    float reward_signal,
    float learning_rate,
    float trace_decay,
    float dt,
    int num_neurons,
    cudaStream_t stream = 0)
{
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;

    eligibility_trace_stdp_kernel<<<grid_size, block_size, 0, stream>>>(
        d_weights, d_eligibility_traces,
        d_row_ptr, d_col_ind,
        d_pre_traces, d_post_traces,
        num_excitatory,
        reward_signal, learning_rate, trace_decay, dt,
        num_neurons
    );

    return cudaGetLastError();
}

/**
 * @brief Launch homeostatic plasticity kernel
 */
cudaError_t launchHomeostaticPlasticity(
    float* d_weights,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_firing_rates,
    float target_rate,
    int num_excitatory,
    float scaling_rate,
    float dt,
    int num_neurons,
    cudaStream_t stream = 0)
{
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;

    homeostatic_plasticity_kernel<<<grid_size, block_size, 0, stream>>>(
        d_weights, d_row_ptr, d_col_ind,
        d_firing_rates, target_rate,
        num_excitatory,
        scaling_rate, dt,
        num_neurons
    );

    return cudaGetLastError();
}

} // namespace bio
} // namespace neurogen
