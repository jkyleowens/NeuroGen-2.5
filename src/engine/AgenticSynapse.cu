#include "engine/AgenticSynapseKernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace neurogen {
namespace engine {

// ============================================================================
// KERNELS
// ============================================================================

/**
 * @brief Updates synaptic weights using the Three-Factor Rule (eProp)
 * 
 * Rule: dW = learning_rate * eligibility_trace * global_error_signal
 * 
 * Eligibility Trace (e_ji):
 * e_ji(t+1) = e_ji(t) * decay + pre_activity * post_derivative
 * 
 * Error Signal (L):
 * We use the "Local Error" from the Column (actual - predicted) as a proxy
 * for the global error signal in this simplified implementation.
 * L = col_state.local_error[col_id]
 */
__global__ void update_synapses_kernel(
    int num_synapses,
    SynapseParams params,
    SynapseState state,
    NeuronState neuron_state,
    cortical::ColumnState col_state,
    cortical::CorticalColumnAgentic* columns,
    int neurons_per_col
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // 1. Load Indices
    int pre_id = state.pre_idx[idx];
    int post_id = state.post_idx[idx];
    
    // 2. Get Pre/Post Activity
    // We use the 'trace' from the neuron state as a smoothed activity measure
    // or 'output' for instantaneous. Let's use 'output' for pre and 'trace' for post-derivative approx.
    float pre_activity = neuron_state.output[pre_id];
    
    // 3. Update Eligibility Trace
    // e_ji = e_ji * decay + pre * post_approx
    // For ALIF, post_derivative is approximated by the surrogate gradient, which is stored in 'output'.
    // So e_ji accumulates when Pre and Post fire together.
    float post_activity = neuron_state.output[post_id];
    
    float e = state.eligibility[idx];
    e = e * params.trace_decay + pre_activity * post_activity;
    state.eligibility[idx] = e;
    
    // 4. Get Error Signal (Learning Signal)
    // We need to find which Column the post-neuron belongs to.
    // post_id -> column_id
    int col_id = post_id / neurons_per_col;
    
    float error_signal = col_state.local_error[col_id];
    
    // 5. Update Weight
    // dW = eta * e * L
    float w = state.weights[idx];
    w += params.learning_rate * e * error_signal;
    
    // 6. Clamp Weight
    if (w > params.w_max) w = params.w_max;
    if (w < params.w_min) w = params.w_min;
    
    state.weights[idx] = w;
    
    // 7. Propagate Input (Forward Pass)
    // Add current to post-synaptic neuron for the NEXT time step
    // I_syn += W * Pre_Output
    if (pre_activity > 0.0f) {
        atomicAdd(&neuron_state.i_syn[post_id], w * pre_activity);
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

void launch_synapse_update(
    int num_synapses,
    SynapseParams params,
    SynapseState state,
    NeuronState neuron_state,
    cortical::ColumnState col_state,
    cortical::CorticalColumnAgentic* columns,
    int neurons_per_col,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (num_synapses + blockSize - 1) / blockSize;
    
    update_synapses_kernel<<<numBlocks, blockSize, 0, stream>>>(
        num_synapses, params, state, neuron_state, col_state, columns, neurons_per_col
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Synapse kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

} // namespace engine
} // namespace neurogen
