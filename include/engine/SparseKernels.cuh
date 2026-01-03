/**
 * @file SparseKernels.cuh
 * @brief CUDA Kernels for Sparse Synaptic Operations
 * 
 * High-performance CUDA kernels for sparse matrix operations:
 * - CSR SpMV for synaptic propagation
 * - BSR block-sparse operations (warp-optimized)
 * - Connectivity generation kernels
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef SPARSE_KERNELS_CUH
#define SPARSE_KERNELS_CUH

#include "SparseSynapseMatrix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace neurogen {
namespace cortical {
namespace kernels {

// ============================================================================
// KERNEL CONFIGURATION
// ============================================================================

constexpr int SPARSE_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = SPARSE_BLOCK_SIZE / WARP_SIZE;

// ============================================================================
// CSR SPARSE MATRIX-VECTOR MULTIPLY (SpMV) - DECLARATIONS
// ============================================================================

/**
 * @brief CSR SpMV kernel for synaptic propagation
 * 
 * Computes: y[post] += sum_pre(W[pre,post] * spike[pre])
 * 
 * This kernel uses one thread per row (postsynaptic neuron) for simplicity.
 * For very sparse matrices, this provides good performance.
 * 
 * @param row_ptr     CSR row pointers [num_post + 1]
 * @param col_idx     CSR column indices [nnz]
 * @param weights     Synaptic weights [nnz]
 * @param spikes      Presynaptic spike vector [num_pre]
 * @param output      Postsynaptic current accumulator [num_post]
 * @param num_post    Number of postsynaptic neurons (rows)
 */
static __global__ void csr_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ weights,
    const float* __restrict__ spikes,
    float* __restrict__ output,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        int pre_idx = col_idx[i];
        float spike_val = spikes[pre_idx];
        if (spike_val > 0.0f) {
            sum += weights[i] * spike_val;
        }
    }
    
    // Accumulate to output (atomic for thread safety if called multiple times)
    atomicAdd(&output[row], sum);
}

/**
 * @brief CSR SpMV with scale factor
 * 
 * Computes: y[post] += scale * sum_pre(W[pre,post] * spike[pre])
 */
__attribute__((unused)) static __global__ void csr_spmv_scaled_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ weights,
    const float* __restrict__ spikes,
    float* __restrict__ output,
    float scale,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        int pre_idx = col_idx[i];
        float spike_val = spikes[pre_idx];
        if (spike_val > 0.0f) {
            sum += weights[i] * spike_val;
        }
    }
    
    atomicAdd(&output[row], sum * scale);
}

/**
 * @brief Warp-parallel CSR SpMV for better load balancing
 * 
 * Uses one warp per row, better for rows with many non-zeros
 */
__attribute__((unused)) static __global__ void csr_spmv_warp_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ weights,
    const float* __restrict__ spikes,
    float* __restrict__ output,
    int num_post
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_post) return;
    
    int start = row_ptr[warp_id];
    int end = row_ptr[warp_id + 1];
    
    float sum = 0.0f;
    
    // Each lane processes elements with stride WARP_SIZE
    for (int i = start + lane_id; i < end; i += WARP_SIZE) {
        int pre_idx = col_idx[i];
        float spike_val = spikes[pre_idx];
        if (spike_val > 0.0f) {
            sum += weights[i] * spike_val;
        }
    }
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Lane 0 writes the result
    if (lane_id == 0) {
        atomicAdd(&output[warp_id], sum);
    }
}

// ============================================================================
// BLOCK SPARSE ROW (BSR) OPERATIONS
// ============================================================================

/**
 * @brief BSR 32x32 block SpMV kernel
 * 
 * Optimized for warp-level parallelism with 32x32 blocks.
 * Each warp processes one row of blocks, threads within warp handle columns.
 */
template<int BLOCK_SIZE = 32>
__attribute__((unused)) static __global__ void bsr_spmv_kernel(
    const int* __restrict__ block_row_ptr,
    const int* __restrict__ block_col_idx,
    const float* __restrict__ block_values,
    const float* __restrict__ block_mask,
    const float* __restrict__ spikes,
    float* __restrict__ output,
    int num_block_rows
) {
    static_assert(BLOCK_SIZE == 32, "BSR kernel optimized for 32x32 blocks");
    
    int block_row = blockIdx.x;
    if (block_row >= num_block_rows) return;
    
    int local_row = threadIdx.x / BLOCK_SIZE;  // Row within block (0-31)
    int local_col = threadIdx.x % BLOCK_SIZE;  // Column within block (0-31)
    
    int block_start = block_row_ptr[block_row];
    int block_end = block_row_ptr[block_row + 1];
    
    float row_sum = 0.0f;
    
    // Iterate over blocks in this block row
    for (int b = block_start; b < block_end; ++b) {
        int block_col = block_col_idx[b];
        
        // Index into dense block
        int block_offset = b * BLOCK_SIZE * BLOCK_SIZE;
        int elem_idx = block_offset + local_row * BLOCK_SIZE + local_col;
        
        // Get weight and mask
        float weight = block_values[elem_idx];
        float mask = block_mask[elem_idx];
        
        // Presynaptic neuron index
        int pre_neuron = block_col * BLOCK_SIZE + local_col;
        
        // Check if presynaptic neuron spiked
        float spike_val = spikes[pre_neuron];
        if (mask > 0.0f && spike_val > 0.0f) {
            row_sum += weight * spike_val;
        }
    }
    
    // Reduce across columns within each row (warp shuffle)
    // Each set of BLOCK_SIZE threads with same local_row needs to sum
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
    }
    
    // Only one thread per row writes result
    if (local_col == 0) {
        int global_row = block_row * BLOCK_SIZE + local_row;
        atomicAdd(&output[global_row], row_sum);
    }
}

/**
 * @brief BSR SpMV with conductance-based synapses
 * 
 * Computes synaptic conductance changes rather than direct currents.
 * g_syn += W * spike * (E_rev - V)
 */
template<int BLOCK_SIZE = 32>
__attribute__((unused)) static __global__ void bsr_conductance_kernel(
    const int* __restrict__ block_row_ptr,
    const int* __restrict__ block_col_idx,
    const float* __restrict__ block_values,
    const float* __restrict__ block_mask,
    const int8_t* __restrict__ synapse_type,  // 0=exc, 1=inh
    const float* __restrict__ spikes,
    const float* __restrict__ voltage,
    float* __restrict__ g_exc,
    float* __restrict__ g_inh,
    float E_exc,
    float E_inh,
    int num_block_rows
) {
    int block_row = blockIdx.x;
    if (block_row >= num_block_rows) return;
    
    int local_row = threadIdx.x / BLOCK_SIZE;
    int local_col = threadIdx.x % BLOCK_SIZE;
    
    int block_start = block_row_ptr[block_row];
    int block_end = block_row_ptr[block_row + 1];
    
    float exc_sum = 0.0f;
    float inh_sum = 0.0f;
    
    for (int b = block_start; b < block_end; ++b) {
        int block_col = block_col_idx[b];
        int block_offset = b * BLOCK_SIZE * BLOCK_SIZE;
        int elem_idx = block_offset + local_row * BLOCK_SIZE + local_col;
        
        float weight = block_values[elem_idx];
        float mask = block_mask[elem_idx];
        int pre_neuron = block_col * BLOCK_SIZE + local_col;
        float spike_val = spikes[pre_neuron];
        
        if (mask > 0.0f && spike_val > 0.0f) {
            int8_t type = synapse_type[elem_idx];
            if (type == 0) {
                exc_sum += weight * spike_val;
            } else {
                inh_sum += weight * spike_val;
            }
        }
    }
    
    // Reduce
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        exc_sum += __shfl_xor_sync(0xffffffff, exc_sum, offset);
        inh_sum += __shfl_xor_sync(0xffffffff, inh_sum, offset);
    }
    
    if (local_col == 0) {
        int global_row = block_row * BLOCK_SIZE + local_row;
        atomicAdd(&g_exc[global_row], exc_sum);
        atomicAdd(&g_inh[global_row], inh_sum);
    }
}

// ============================================================================
// TRACE UPDATE KERNELS
// ============================================================================

/**
 * @brief Update presynaptic traces based on spikes
 * 
 * trace[i] = decay * trace[i] + (spike[i] ? 1.0 : 0.0)
 */
__attribute__((unused)) static __global__ void update_pre_trace_kernel(
    float* __restrict__ trace,
    const float* __restrict__ spikes,
    float decay,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    float tr = trace[idx] * decay;
    if (spikes[idx] > 0.0f) {
        tr += 1.0f; // Binary trace update for now, or use spikes[idx] for weighted trace
    }
    trace[idx] = tr;
}

/**
 * @brief Update postsynaptic traces based on spikes
 */
__attribute__((unused)) static __global__ void update_post_trace_kernel(
    float* __restrict__ trace,
    const float* __restrict__ spikes,
    float decay,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    float tr = trace[idx] * decay;
    if (spikes[idx] > 0.0f) {
        tr += 1.0f;
    }
    trace[idx] = tr;
}

// ============================================================================
// CONNECTIVITY GENERATION KERNELS
// ============================================================================

/**
 * @brief Initialize random states for connectivity generation
 */
__attribute__((unused)) static __global__ void init_random_states_kernel(
    curandState* __restrict__ states,
    unsigned long long seed,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * @brief Count connections per row for CSR allocation
 * 
 * Random connectivity with probability p
 */
__attribute__((unused)) static __global__ void count_connections_random_kernel(
    int* __restrict__ row_counts,
    curandState* __restrict__ states,
    float probability,
    int num_pre,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    curandState local_state = states[row];
    int count = 0;
    
    for (int col = 0; col < num_pre; ++col) {
        float r = curand_uniform(&local_state);
        if (r < probability) {
            count++;
        }
    }
    
    row_counts[row] = count;
    states[row] = local_state;
}

/**
 * @brief Generate random connectivity (fill CSR structure)
 */
__attribute__((unused)) static __global__ void generate_random_connectivity_kernel(
    const int* __restrict__ row_ptr,
    int* __restrict__ col_idx,
    float* __restrict__ weights,
    curandState* __restrict__ states,
    float probability,
    float w_init_mean,
    float w_init_std,
    int num_pre,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    curandState local_state = states[row];
    int write_idx = row_ptr[row];
    
    for (int col = 0; col < num_pre; ++col) {
        float r = curand_uniform(&local_state);
        if (r < probability) {
            col_idx[write_idx] = col;
            // Initialize weight with Gaussian distribution
            float w = w_init_mean + w_init_std * curand_normal(&local_state);
            w = fmaxf(0.0f, fminf(1.0f, w));  // Clip to [0, 1]
            weights[write_idx] = w;
            write_idx++;
        }
    }
    
    states[row] = local_state;
}

/**
 * @brief Generate distance-dependent connectivity
 * 
 * P(connection) = exp(-distance^2 / (2 * sigma^2))
 */
__attribute__((unused)) static __global__ void generate_distance_connectivity_kernel(
    const int* __restrict__ row_ptr,
    int* __restrict__ col_idx,
    float* __restrict__ weights,
    const float* __restrict__ pre_positions,   // [num_pre, 2] or [num_pre, 3]
    const float* __restrict__ post_positions,  // [num_post, 2] or [num_post, 3]
    curandState* __restrict__ states,
    float sigma,
    float base_probability,
    float w_init_mean,
    float w_init_std,
    int num_pre,
    int num_post,
    int dims
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    curandState local_state = states[row];
    int write_idx = row_ptr[row];
    
    // Get post neuron position
    float post_pos[3] = {0.0f, 0.0f, 0.0f};
    for (int d = 0; d < dims; ++d) {
        post_pos[d] = post_positions[row * dims + d];
    }
    
    float sigma_sq_2 = 2.0f * sigma * sigma;
    
    for (int col = 0; col < num_pre; ++col) {
        // Compute distance
        float dist_sq = 0.0f;
        for (int d = 0; d < dims; ++d) {
            float diff = pre_positions[col * dims + d] - post_pos[d];
            dist_sq += diff * diff;
        }
        
        // Distance-dependent probability
        float p = base_probability * __expf(-dist_sq / sigma_sq_2);
        
        float r = curand_uniform(&local_state);
        if (r < p) {
            col_idx[write_idx] = col;
            float w = w_init_mean + w_init_std * curand_normal(&local_state);
            w = fmaxf(0.0f, fminf(1.0f, w));
            weights[write_idx] = w;
            write_idx++;
        }
    }
    
    states[row] = local_state;
}

/**
 * @brief Initialize eligibility traces to zero
 */
__attribute__((unused)) static __global__ void init_eligibility_kernel(
    float* __restrict__ eligibility,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    eligibility[idx] = 0.0f;
}

// ============================================================================
// WEIGHT OPERATIONS
// ============================================================================

/**
 * @brief Clip weights to [w_min, w_max]
 */
__attribute__((unused)) static __global__ void clip_weights_kernel(
    float* __restrict__ weights,
    float w_min,
    float w_max,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    
    float w = weights[idx];
    weights[idx] = fmaxf(w_min, fminf(w_max, w));
}

/**
 * @brief Apply weight decay
 */
__attribute__((unused)) static __global__ void weight_decay_kernel(
    float* __restrict__ weights,
    float decay_factor,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    
    weights[idx] *= decay_factor;
}

/**
 * @brief Normalize weights per postsynaptic neuron
 */
__attribute__((unused)) static __global__ void normalize_weights_per_post_kernel(
    const int* __restrict__ row_ptr,
    float* __restrict__ weights,
    float target_sum,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    if (start == end) return;  // No connections
    
    // Compute sum
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        sum += weights[i];
    }
    
    if (sum > 0.0f) {
        float scale = target_sum / sum;
        for (int i = start; i < end; ++i) {
            weights[i] *= scale;
        }
    }
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Launch CSR SpMV kernel
 */
inline cudaError_t launchCSRSpMV(
    const CSRSynapseMatrix& synapses,
    const float* d_spikes,
    float* d_output,
    float scale,
    cudaStream_t stream = 0
) {
    int grid_size = (synapses.num_post + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
    
    if (scale == 1.0f) {
        csr_spmv_kernel<<<grid_size, SPARSE_BLOCK_SIZE, 0, stream>>>(
            synapses.d_row_ptr,
            synapses.d_col_idx,
            synapses.d_weights,
            d_spikes,
            d_output,
            synapses.num_post
        );
    } else {
        csr_spmv_scaled_kernel<<<grid_size, SPARSE_BLOCK_SIZE, 0, stream>>>(
            synapses.d_row_ptr,
            synapses.d_col_idx,
            synapses.d_weights,
            d_spikes,
            d_output,
            scale,
            synapses.num_post
        );
    }
    
    return cudaGetLastError();
}

/**
 * @brief Launch warp-parallel CSR SpMV (for denser matrices)
 */
inline cudaError_t launchCSRSpMVWarp(
    const CSRSynapseMatrix& synapses,
    const float* d_spikes,
    float* d_output,
    cudaStream_t stream = 0
) {
    // One warp per row
    int num_warps = synapses.num_post;
    int threads_needed = num_warps * WARP_SIZE;
    int grid_size = (threads_needed + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
    
    csr_spmv_warp_kernel<<<grid_size, SPARSE_BLOCK_SIZE, 0, stream>>>(
        synapses.d_row_ptr,
        synapses.d_col_idx,
        synapses.d_weights,
        d_spikes,
        d_output,
        synapses.num_post
    );
    
    return cudaGetLastError();
}

/**
 * @brief Launch BSR SpMV kernel
 */
template<int BLOCK_SIZE = 32>
inline cudaError_t launchBSRSpMV(
    const BSRSynapseMatrix<BLOCK_SIZE>& synapses,
    const float* d_spikes,
    float* d_output,
    cudaStream_t stream = 0
) {
    // One block per block-row, threads handle the dense block
    int grid_size = synapses.num_block_rows;
    int block_size = BLOCK_SIZE * BLOCK_SIZE;  // 1024 threads for 32x32
    
    // Note: May need to limit to 1024 threads per block
    if (block_size > 1024) {
        // Fall back to smaller processing
        return cudaErrorInvalidConfiguration;
    }
    
    bsr_spmv_kernel<BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
        synapses.d_block_row_ptr,
        synapses.d_block_col_idx,
        synapses.d_block_values,
        synapses.d_block_mask,
        d_spikes,
        d_output,
        synapses.num_block_rows
    );
    
    return cudaGetLastError();
}

/**
 * @brief Update presynaptic traces
 */
inline cudaError_t launchUpdatePreTrace(
    float* d_trace,
    const float* d_spikes,
    float tau,
    float dt,
    int num_neurons,
    cudaStream_t stream = 0
) {
    float decay = expf(-dt / tau);
    int grid_size = (num_neurons + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
    
    update_pre_trace_kernel<<<grid_size, SPARSE_BLOCK_SIZE, 0, stream>>>(
        d_trace, d_spikes, decay, num_neurons
    );
    
    return cudaGetLastError();
}

/**
 * @brief Update postsynaptic traces
 */
inline cudaError_t launchUpdatePostTrace(
    float* d_trace,
    const float* d_spikes,
    float tau,
    float dt,
    int num_neurons,
    cudaStream_t stream = 0
) {
    float decay = expf(-dt / tau);
    int grid_size = (num_neurons + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
    
    update_post_trace_kernel<<<grid_size, SPARSE_BLOCK_SIZE, 0, stream>>>(
        d_trace, d_spikes, decay, num_neurons
    );
    
    return cudaGetLastError();
}

/**
 * @brief Clip weights
 */
inline cudaError_t launchClipWeights(
    float* d_weights,
    float w_min,
    float w_max,
    int nnz,
    cudaStream_t stream = 0
) {
    int grid_size = (nnz + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE;
    
    clip_weights_kernel<<<grid_size, SPARSE_BLOCK_SIZE, 0, stream>>>(
        d_weights, w_min, w_max, nnz
    );
    
    return cudaGetLastError();
}

} // namespace kernels
} // namespace cortical
} // namespace neurogen

#endif // SPARSE_KERNELS_CUH
