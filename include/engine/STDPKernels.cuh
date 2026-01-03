/**
 * @file STDPKernels.cuh
 * @brief CUDA Kernels for Spike-Timing-Dependent Plasticity (STDP)
 * 
 * Implements various STDP rules:
 * - Classic asymmetric STDP (pair-based)
 * - Triplet STDP (Pfister & Gerstner, 2006)
 * - Eligibility trace STDP for reinforcement learning
 * - Calcium-gated plasticity
 * - Voltage-dependent plasticity
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef STDP_KERNELS_CUH
#define STDP_KERNELS_CUH

#include "SparseSynapseMatrix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace neurogen {
namespace cortical {
namespace kernels {

// ============================================================================
// STDP PARAMETERS
// ============================================================================

/**
 * @brief Parameters for STDP learning
 */
struct STDPParams {
    // Time constants
    float tau_pre = 20.0f;          // Presynaptic trace decay (ms)
    float tau_post = 20.0f;         // Postsynaptic trace decay (ms)
    float tau_slow = 100.0f;        // Slow trace for triplet rule (ms)
    float tau_eligibility = 1000.0f; // Eligibility trace decay (ms)
    float tau_calcium = 50.0f;       // Calcium decay (ms)
    
    // Learning amplitudes
    float A_plus = 0.01f;           // LTP amplitude
    float A_minus = 0.012f;         // LTD amplitude (slightly stronger for stability)
    float A_triplet = 0.005f;       // Triplet term amplitude
    
    // Weight bounds
    float w_min = 0.0f;
    float w_max = 1.0f;
    
    // Calcium thresholds
    float theta_d = 1.0f;           // LTD calcium threshold
    float theta_p = 1.3f;           // LTP calcium threshold
    
    // Metaplasticity
    float metaplasticity_rate = 0.001f;  // BCM threshold adaptation rate
    
    // Global modulation
    float learning_rate = 1.0f;     // Global learning rate multiplier
};

// ============================================================================
// CLASSIC STDP (PAIR-BASED)
// ============================================================================

/**
 * @brief Classic asymmetric STDP kernel (CSR format)
 * 
 * When post spikes: dw = A_plus * trace_pre (LTP)
 * When pre spikes:  dw = -A_minus * trace_post (LTD)
 * 
 * This kernel updates weights based on spike timing.
 */
__attribute__((unused)) static __global__ void stdp_classic_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ weights,
    float* __restrict__ eligibility,
    const float* __restrict__ pre_trace,
    const float* __restrict__ post_trace,
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    const STDPParams params,
    float dt,
    int num_post
) {
    int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= num_post) return;
    
    int start = row_ptr[post];
    int end = row_ptr[post + 1];
    
    bool post_spike = (post_spikes[post] > 0.0f);
    float post_tr = post_trace[post];
    
    for (int i = start; i < end; ++i) {
        int pre = col_idx[i];
        bool pre_spike = (pre_spikes[pre] > 0.0f);
        float pre_tr = pre_trace[pre];
        
        float dw = 0.0f;
        
        // LTP: post spike, use pre trace
        if (post_spike) {
            dw += params.A_plus * pre_tr;
        }
        
        // LTD: pre spike, use post trace  
        if (pre_spike) {
            dw -= params.A_minus * post_tr;
        }
        
        // Scale by learning rate
        dw *= params.learning_rate;
        
        // Update weight
        float w = weights[i] + dw;
        weights[i] = fmaxf(params.w_min, fminf(params.w_max, w));
        
        // Update eligibility trace
        float decay = __expf(-dt / params.tau_eligibility);
        eligibility[i] = eligibility[i] * decay + fabsf(dw);
    }
}

// ============================================================================
// TRIPLET STDP (Pfister & Gerstner, 2006)
// ============================================================================

/**
 * @brief Triplet STDP kernel
 * 
 * Includes a slow postsynaptic trace for triplet interactions.
 * This captures frequency-dependent effects better than pair-based STDP.
 */
__attribute__((unused)) static __global__ void stdp_triplet_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ weights,
    float* __restrict__ eligibility,
    const float* __restrict__ pre_trace,
    const float* __restrict__ post_trace,
    const float* __restrict__ post_trace_slow,  // Slow postsynaptic trace
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    const STDPParams params,
    float dt,
    int num_post
) {
    int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= num_post) return;
    
    int start = row_ptr[post];
    int end = row_ptr[post + 1];
    
    bool post_spike = (post_spikes[post] > 0.0f);
    float post_tr = post_trace[post];
    float post_tr_slow = post_trace_slow[post];
    
    for (int i = start; i < end; ++i) {
        int pre = col_idx[i];
        bool pre_spike = (pre_spikes[pre] > 0.0f);
        float pre_tr = pre_trace[pre];
        
        float dw = 0.0f;
        
        // Pair-based terms
        if (post_spike) {
            // LTP with triplet enhancement
            dw += params.A_plus * pre_tr * (1.0f + params.A_triplet * post_tr_slow);
        }
        
        if (pre_spike) {
            // LTD 
            dw -= params.A_minus * post_tr;
        }
        
        dw *= params.learning_rate;
        
        float w = weights[i] + dw;
        weights[i] = fmaxf(params.w_min, fminf(params.w_max, w));
        
        float decay = __expf(-dt / params.tau_eligibility);
        eligibility[i] = eligibility[i] * decay + fabsf(dw);
    }
}

// ============================================================================
// ELIGIBILITY-TRACE STDP FOR REINFORCEMENT LEARNING
// ============================================================================

/**
 * @brief Update eligibility traces based on STDP events
 * 
 * Eligibility trace captures "credit assignment" for RL.
 * Actual weight updates happen when reward signal arrives.
 */
__attribute__((unused)) static __global__ void stdp_eligibility_update_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ eligibility,
    const float* __restrict__ pre_trace,
    const float* __restrict__ post_trace,
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    const STDPParams params,
    float dt,
    int num_post
) {
    int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= num_post) return;
    
    int start = row_ptr[post];
    int end = row_ptr[post + 1];
    
    bool post_spike = (post_spikes[post] > 0.0f);
    float post_tr = post_trace[post];
    float decay = __expf(-dt / params.tau_eligibility);
    
    for (int i = start; i < end; ++i) {
        int pre = col_idx[i];
        bool pre_spike = (pre_spikes[pre] > 0.0f);
        float pre_tr = pre_trace[pre];
        
        // Decay existing eligibility
        float e = eligibility[i] * decay;
        
        // Add STDP-based eligibility
        if (post_spike) {
            e += params.A_plus * pre_tr;
        }
        if (pre_spike) {
            e -= params.A_minus * post_tr;
        }
        
        eligibility[i] = e;
    }
}

/**
 * @brief Apply reward-modulated weight updates
 * 
 * dw = learning_rate * reward * eligibility
 */
__attribute__((unused)) static __global__ void stdp_reward_modulation_kernel(
    float* __restrict__ weights,
    const float* __restrict__ eligibility,
    float reward,           // Scalar reward signal (dopamine)
    float learning_rate,
    float w_min,
    float w_max,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    
    float dw = learning_rate * reward * eligibility[idx];
    float w = weights[idx] + dw;
    weights[idx] = fmaxf(w_min, fminf(w_max, w));
}

/**
 * @brief Three-factor learning rule with neuromodulation
 * 
 * dw = learning_rate * eligibility * (dopamine * alpha_d + acetylcholine * alpha_a)
 */
__attribute__((unused)) static __global__ void stdp_neuromodulated_kernel(
    float* __restrict__ weights,
    const float* __restrict__ eligibility,
    float dopamine,
    float acetylcholine,
    float serotonin,
    float alpha_dopamine,    // Dopamine sensitivity
    float alpha_ach,         // ACh sensitivity
    float alpha_serotonin,   // 5-HT sensitivity
    float learning_rate,
    float w_min,
    float w_max,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    
    // Combined neuromodulatory signal
    float modulation = dopamine * alpha_dopamine 
                     + acetylcholine * alpha_ach
                     + serotonin * alpha_serotonin;
    
    float dw = learning_rate * eligibility[idx] * modulation;
    float w = weights[idx] + dw;
    weights[idx] = fmaxf(w_min, fminf(w_max, w));
}

// ============================================================================
// CALCIUM-GATED PLASTICITY
// ============================================================================

/**
 * @brief Calcium-based STDP kernel
 * 
 * Uses calcium concentration to gate LTP/LTD:
 * - Low calcium: no change
 * - Medium calcium (theta_d < Ca < theta_p): LTD
 * - High calcium (Ca > theta_p): LTP
 */
__attribute__((unused)) static __global__ void stdp_calcium_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ weights,
    const float* __restrict__ pre_calcium,
    const float* __restrict__ post_calcium,
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    const STDPParams params,
    int num_post
) {
    int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= num_post) return;
    
    int start = row_ptr[post];
    int end = row_ptr[post + 1];
    
    float ca_post = post_calcium[post];
    bool post_spike = (post_spikes[post] > 0.0f);
    
    for (int i = start; i < end; ++i) {
        int pre = col_idx[i];
        float ca_pre = pre_calcium[pre];
        
        // Combined calcium at synapse (simplified)
        float ca_syn = 0.5f * (ca_pre + ca_post);
        
        // Calcium-gated plasticity
        float dw = 0.0f;
        
        if (ca_syn > params.theta_p) {
            // LTP zone
            dw = params.A_plus * params.learning_rate;
        } else if (ca_syn > params.theta_d) {
            // LTD zone
            dw = -params.A_minus * params.learning_rate * 
                 ((ca_syn - params.theta_d) / (params.theta_p - params.theta_d));
        }
        // Below theta_d: no plasticity
        
        float w = weights[i] + dw;
        weights[i] = fmaxf(params.w_min, fminf(params.w_max, w));
    }
}

/**
 * @brief Update calcium traces
 */
__attribute__((unused)) static __global__ void update_calcium_kernel(
    float* __restrict__ calcium,
    const float* __restrict__ spikes,
    float calcium_increment,
    float decay,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    float ca = calcium[idx] * decay;
    if (spikes[idx] > 0.0f) {
        ca += calcium_increment;
    }
    calcium[idx] = ca;
}

// ============================================================================
// VOLTAGE-DEPENDENT STDP
// ============================================================================

/**
 * @brief Voltage-dependent STDP (based on Clopath et al., 2010)
 * 
 * LTP requires depolarization above threshold
 * LTD is proportional to low-pass filtered voltage
 */
__attribute__((unused)) static __global__ void stdp_voltage_dependent_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ weights,
    const float* __restrict__ pre_trace,
    const float* __restrict__ post_voltage,
    const float* __restrict__ post_voltage_filtered,  // Low-pass filtered
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    float theta_minus,      // LTD voltage threshold
    float theta_plus,       // LTP voltage threshold
    const STDPParams params,
    int num_post
) {
    int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= num_post) return;
    
    int start = row_ptr[post];
    int end = row_ptr[post + 1];
    
    float v_post = post_voltage[post];
    float v_filtered = post_voltage_filtered[post];
    bool post_spike = (post_spikes[post] > 0.0f);
    
    for (int i = start; i < end; ++i) {
        int pre = col_idx[i];
        bool pre_spike = (pre_spikes[pre] > 0.0f);
        float pre_tr = pre_trace[pre];
        
        float dw = 0.0f;
        
        // LTD: depends on filtered voltage above threshold
        if (pre_spike && v_filtered > theta_minus) {
            dw -= params.A_minus * (v_filtered - theta_minus);
        }
        
        // LTP: requires post spike AND pre trace AND voltage above threshold
        if (post_spike && v_post > theta_plus) {
            dw += params.A_plus * pre_tr * (v_post - theta_plus);
        }
        
        dw *= params.learning_rate;
        
        float w = weights[i] + dw;
        weights[i] = fmaxf(params.w_min, fminf(params.w_max, w));
    }
}

// ============================================================================
// HOMEOSTATIC PLASTICITY
// ============================================================================

/**
 * @brief Synaptic scaling for homeostasis
 * 
 * Scale weights to maintain target firing rate
 */
__attribute__((unused)) static __global__ void homeostatic_scaling_kernel(
    const int* __restrict__ row_ptr,
    float* __restrict__ weights,
    const float* __restrict__ firing_rate,
    float target_rate,
    float scaling_rate,
    float w_min,
    float w_max,
    int num_post
) {
    int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= num_post) return;
    
    float fr = firing_rate[post];
    
    // Scaling factor based on deviation from target
    // scale > 1 if below target, < 1 if above
    float scale = 1.0f + scaling_rate * (target_rate - fr) / target_rate;
    scale = fmaxf(0.5f, fminf(2.0f, scale));  // Limit scaling range
    
    int start = row_ptr[post];
    int end = row_ptr[post + 1];
    
    for (int i = start; i < end; ++i) {
        float w = weights[i] * scale;
        weights[i] = fmaxf(w_min, fminf(w_max, w));
    }
}

/**
 * @brief Intrinsic plasticity (threshold adaptation)
 */
__attribute__((unused)) static __global__ void intrinsic_plasticity_kernel(
    float* __restrict__ threshold,
    const float* __restrict__ firing_rate,
    float target_rate,
    float adaptation_rate,
    float theta_min,
    float theta_max,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    float fr = firing_rate[idx];
    
    // Increase threshold if firing too fast, decrease if too slow
    float d_theta = adaptation_rate * (fr - target_rate);
    
    float theta = threshold[idx] + d_theta;
    threshold[idx] = fmaxf(theta_min, fminf(theta_max, theta));
}

/**
 * @brief BCM-style metaplasticity (sliding threshold)
 */
__attribute__((unused)) static __global__ void bcm_metaplasticity_kernel(
    float* __restrict__ bcm_threshold,
    const float* __restrict__ firing_rate,
    const float* __restrict__ firing_rate_avg,  // Long-term average
    float adaptation_rate,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // BCM threshold tracks squared average firing rate
    float fr_avg = firing_rate_avg[idx];
    float theta_target = fr_avg * fr_avg;
    
    float theta = bcm_threshold[idx];
    theta += adaptation_rate * (theta_target - theta);
    bcm_threshold[idx] = fmaxf(0.01f, theta);  // Prevent going to zero
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

constexpr int STDP_BLOCK_SIZE = 256;

/**
 * @brief Launch classic STDP
 */
inline cudaError_t launchSTDPClassic(
    CSRSynapseMatrix& synapses,
    const float* d_pre_spikes,
    const float* d_post_spikes,
    const STDPParams& params,
    float dt,
    cudaStream_t stream = 0
) {
    int grid_size = (synapses.num_post + STDP_BLOCK_SIZE - 1) / STDP_BLOCK_SIZE;
    
    stdp_classic_kernel<<<grid_size, STDP_BLOCK_SIZE, 0, stream>>>(
        synapses.d_row_ptr,
        synapses.d_col_idx,
        synapses.d_weights,
        synapses.d_eligibility,
        synapses.d_pre_trace,
        synapses.d_post_trace,
        d_pre_spikes,
        d_post_spikes,
        params,
        dt,
        synapses.num_post
    );
    
    return cudaGetLastError();
}

/**
 * @brief Launch eligibility trace update
 */
inline cudaError_t launchEligibilityUpdate(
    CSRSynapseMatrix& synapses,
    const float* d_pre_spikes,
    const float* d_post_spikes,
    const STDPParams& params,
    float dt,
    cudaStream_t stream = 0
) {
    int grid_size = (synapses.num_post + STDP_BLOCK_SIZE - 1) / STDP_BLOCK_SIZE;
    
    stdp_eligibility_update_kernel<<<grid_size, STDP_BLOCK_SIZE, 0, stream>>>(
        synapses.d_row_ptr,
        synapses.d_col_idx,
        synapses.d_eligibility,
        synapses.d_pre_trace,
        synapses.d_post_trace,
        d_pre_spikes,
        d_post_spikes,
        params,
        dt,
        synapses.num_post
    );
    
    return cudaGetLastError();
}

/**
 * @brief Apply reward-modulated weight updates
 */
inline cudaError_t launchRewardModulation(
    CSRSynapseMatrix& synapses,
    float reward,
    float learning_rate,
    cudaStream_t stream = 0
) {
    int grid_size = (synapses.nnz + STDP_BLOCK_SIZE - 1) / STDP_BLOCK_SIZE;
    
    stdp_reward_modulation_kernel<<<grid_size, STDP_BLOCK_SIZE, 0, stream>>>(
        synapses.d_weights,
        synapses.d_eligibility,
        reward,
        learning_rate,
        0.0f,  // w_min
        1.0f,  // w_max
        synapses.nnz
    );
    
    return cudaGetLastError();
}

/**
 * @brief Launch neuromodulated STDP
 */
inline cudaError_t launchNeuromodulatedSTDP(
    CSRSynapseMatrix& synapses,
    float dopamine,
    float acetylcholine,
    float serotonin,
    float learning_rate,
    cudaStream_t stream = 0
) {
    int grid_size = (synapses.nnz + STDP_BLOCK_SIZE - 1) / STDP_BLOCK_SIZE;
    
    // Default sensitivities
    const float alpha_d = 1.0f;
    const float alpha_a = 0.5f;
    const float alpha_s = 0.3f;
    
    stdp_neuromodulated_kernel<<<grid_size, STDP_BLOCK_SIZE, 0, stream>>>(
        synapses.d_weights,
        synapses.d_eligibility,
        dopamine,
        acetylcholine,
        serotonin,
        alpha_d,
        alpha_a,
        alpha_s,
        learning_rate,
        0.0f,  // w_min
        1.0f,  // w_max
        synapses.nnz
    );
    
    return cudaGetLastError();
}

/**
 * @brief Launch homeostatic scaling
 */
inline cudaError_t launchHomeostaticScaling(
    CSRSynapseMatrix& synapses,
    const float* d_firing_rate,
    float target_rate,
    float scaling_rate,
    cudaStream_t stream = 0
) {
    int grid_size = (synapses.num_post + STDP_BLOCK_SIZE - 1) / STDP_BLOCK_SIZE;
    
    homeostatic_scaling_kernel<<<grid_size, STDP_BLOCK_SIZE, 0, stream>>>(
        synapses.d_row_ptr,
        synapses.d_weights,
        d_firing_rate,
        target_rate,
        scaling_rate,
        0.0f,  // w_min
        1.0f,  // w_max
        synapses.num_post
    );
    
    return cudaGetLastError();
}

/**
 * @brief Update calcium traces
 */
inline cudaError_t launchCalciumUpdate(
    float* d_calcium,
    const float* d_spikes,
    float calcium_increment,
    float tau_calcium,
    float dt,
    int num_neurons,
    cudaStream_t stream = 0
) {
    int grid_size = (num_neurons + STDP_BLOCK_SIZE - 1) / STDP_BLOCK_SIZE;
    float decay = expf(-dt / tau_calcium);
    
    update_calcium_kernel<<<grid_size, STDP_BLOCK_SIZE, 0, stream>>>(
        d_calcium, d_spikes, calcium_increment, decay, num_neurons
    );
    
    return cudaGetLastError();
}

} // namespace kernels
} // namespace cortical
} // namespace neurogen

#endif // STDP_KERNELS_CUH
