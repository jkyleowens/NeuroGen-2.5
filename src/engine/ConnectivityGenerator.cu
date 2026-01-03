/**
 * @file ConnectivityGenerator.cu
 * @brief GPU-accelerated connectivity pattern generation
 * 
 * Generates biologically-inspired connectivity patterns:
 * - Random (Erdős–Rényi)
 * - Small-world (Watts-Strogatz)
 * - Distance-dependent (Gaussian decay)
 * - Layer-specific (canonical cortical)
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#include "engine/SparseSynapseMatrix.h"
#include "engine/SparseKernels.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <random>
#include <algorithm>

namespace neurogen {
namespace cortical {

// ============================================================================
// HOST-SIDE CONNECTIVITY GENERATION
// ============================================================================

/**
 * @brief Generate random connectivity (Erdős–Rényi model)
 */
cudaError_t generateRandomConnectivity(
    CSRSynapseMatrix& matrix,
    int num_pre,
    int num_post,
    float probability,
    float w_mean,
    float w_std,
    unsigned int seed
) {
    // Generate on CPU (for correctness), then upload to GPU
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(w_mean, w_std);
    
    std::vector<int> row_ptr(num_post + 1);
    std::vector<int> col_idx;
    std::vector<float> weights;
    
    row_ptr[0] = 0;
    
    for (int post = 0; post < num_post; ++post) {
        for (int pre = 0; pre < num_pre; ++pre) {
            if (uniform(rng) < probability) {
                col_idx.push_back(pre);
                float w = normal(rng);
                w = std::max(0.0f, std::min(1.0f, w));
                weights.push_back(w);
            }
        }
        row_ptr[post + 1] = col_idx.size();
    }
    
    int nnz = col_idx.size();
    
    // Allocate GPU memory
    cudaError_t err = matrix.allocate(num_pre, num_post, nnz);
    if (err != cudaSuccess) return err;
    
    // Upload to GPU
    err = cudaMemcpy(matrix.d_row_ptr, row_ptr.data(), (num_post + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(matrix.d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(matrix.d_weights, weights.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    // Initialize plasticity arrays to zero
    err = cudaMemset(matrix.d_eligibility, 0, nnz * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(matrix.d_pre_trace, 0, num_pre * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(matrix.d_post_trace, 0, num_post * sizeof(float));
    if (err != cudaSuccess) return err;
    
    return cudaSuccess;
}

/**
 * @brief Generate small-world connectivity (Watts-Strogatz model)
 */
cudaError_t generateSmallWorldConnectivity(
    CSRSynapseMatrix& matrix,
    int num_pre,
    int num_post,
    int k_neighbors,        // Number of nearest neighbors
    float rewire_prob,      // Rewiring probability
    float w_mean,
    float w_std,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::uniform_int_distribution<int> rand_node(0, num_pre - 1);
    std::normal_distribution<float> normal(w_mean, w_std);
    
    // Start with ring lattice
    std::vector<std::vector<std::pair<int, float>>> adj(num_post);
    
    for (int i = 0; i < num_post; ++i) {
        for (int j = 1; j <= k_neighbors / 2; ++j) {
            // Connect to k/2 neighbors on each side (wrapping)
            int left = (i - j + num_pre) % num_pre;
            int right = (i + j) % num_pre;
            
            float w = normal(rng);
            w = std::max(0.0f, std::min(1.0f, w));
            
            adj[i].push_back({left, w});
            if (left != right) {
                w = normal(rng);
                w = std::max(0.0f, std::min(1.0f, w));
                adj[i].push_back({right, w});
            }
        }
    }
    
    // Rewire edges with probability p
    for (int i = 0; i < num_post; ++i) {
        for (auto& edge : adj[i]) {
            if (uniform(rng) < rewire_prob) {
                // Rewire to random node
                int new_target;
                do {
                    new_target = rand_node(rng);
                } while (new_target == i);  // Avoid self-connections
                
                edge.first = new_target;
            }
        }
        // Sort by target for CSR format
        std::sort(adj[i].begin(), adj[i].end());
    }
    
    // Convert to CSR
    std::vector<int> row_ptr(num_post + 1);
    std::vector<int> col_idx;
    std::vector<float> weights;
    
    row_ptr[0] = 0;
    for (int i = 0; i < num_post; ++i) {
        for (const auto& edge : adj[i]) {
            col_idx.push_back(edge.first);
            weights.push_back(edge.second);
        }
        row_ptr[i + 1] = col_idx.size();
    }
    
    int nnz = col_idx.size();
    
    cudaError_t err = matrix.allocate(num_pre, num_post, nnz);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(matrix.d_row_ptr, row_ptr.data(), (num_post + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(matrix.d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(matrix.d_weights, weights.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(matrix.d_eligibility, 0, nnz * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMemset(matrix.d_pre_trace, 0, num_pre * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMemset(matrix.d_post_trace, 0, num_post * sizeof(float));
    
    return err;
}

/**
 * @brief Generate distance-dependent connectivity
 * 
 * Connection probability decays with distance: P(d) = p_base * exp(-d^2 / (2*sigma^2))
 */
cudaError_t generateDistanceDependentConnectivity(
    CSRSynapseMatrix& matrix,
    int num_pre,
    int num_post,
    const std::vector<float>& pre_positions,   // [num_pre * dims]
    const std::vector<float>& post_positions,  // [num_post * dims]
    int dims,               // 2D or 3D
    float sigma,            // Distance scale
    float p_base,           // Base probability
    float w_mean,
    float w_std,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(w_mean, w_std);
    
    std::vector<int> row_ptr(num_post + 1);
    std::vector<int> col_idx;
    std::vector<float> weights;
    
    float sigma_sq_2 = 2.0f * sigma * sigma;
    
    row_ptr[0] = 0;
    for (int post = 0; post < num_post; ++post) {
        for (int pre = 0; pre < num_pre; ++pre) {
            // Compute distance
            float dist_sq = 0.0f;
            for (int d = 0; d < dims; ++d) {
                float diff = pre_positions[pre * dims + d] - post_positions[post * dims + d];
                dist_sq += diff * diff;
            }
            
            // Distance-dependent probability
            float prob = p_base * std::exp(-dist_sq / sigma_sq_2);
            
            if (uniform(rng) < prob) {
                col_idx.push_back(pre);
                float w = normal(rng);
                w = std::max(0.0f, std::min(1.0f, w));
                weights.push_back(w);
            }
        }
        row_ptr[post + 1] = col_idx.size();
    }
    
    int nnz = col_idx.size();
    
    cudaError_t err = matrix.allocate(num_pre, num_post, nnz);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(matrix.d_row_ptr, row_ptr.data(), (num_post + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(matrix.d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(matrix.d_weights, weights.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(matrix.d_eligibility, 0, nnz * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMemset(matrix.d_pre_trace, 0, num_pre * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMemset(matrix.d_post_trace, 0, num_post * sizeof(float));
    
    return err;
}

/**
 * @brief Generate canonical cortical connectivity between layers
 */
cudaError_t generateInterLayerConnectivity(
    CSRSynapseMatrix& matrix,
    int source_layer,       // 0-5 for L1-L6
    int target_layer,       // 0-5 for L1-L6
    int num_source,
    int num_target,
    float exc_ratio,        // Fraction of excitatory connections
    float w_exc_mean,
    float w_inh_mean,
    unsigned int seed
) {
    // Use canonical connectivity probabilities
    static const float canonical_conn[6][6] = {
        // To:  L1    L2    L3    L4    L5    L6
        /*L1*/ {0.00f, 0.02f, 0.02f, 0.00f, 0.01f, 0.01f},
        /*L2*/ {0.05f, 0.10f, 0.15f, 0.02f, 0.08f, 0.05f},
        /*L3*/ {0.05f, 0.15f, 0.12f, 0.02f, 0.10f, 0.08f},
        /*L4*/ {0.01f, 0.20f, 0.15f, 0.08f, 0.05f, 0.10f},
        /*L5*/ {0.02f, 0.05f, 0.08f, 0.02f, 0.10f, 0.15f},
        /*L6*/ {0.02f, 0.03f, 0.05f, 0.15f, 0.08f, 0.08f}
    };
    
    float prob = canonical_conn[source_layer][target_layer];
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    
    std::vector<int> row_ptr(num_target + 1);
    std::vector<int> col_idx;
    std::vector<float> weights;
    
    row_ptr[0] = 0;
    for (int target = 0; target < num_target; ++target) {
        for (int source = 0; source < num_source; ++source) {
            if (uniform(rng) < prob) {
                col_idx.push_back(source);
                
                // Determine if excitatory or inhibitory
                bool is_exc = uniform(rng) < exc_ratio;
                float w = is_exc ? w_exc_mean : -w_inh_mean;
                // Add some variance
                w *= (0.8f + 0.4f * uniform(rng));
                weights.push_back(w);
            }
        }
        row_ptr[target + 1] = col_idx.size();
    }
    
    int nnz = col_idx.size();
    if (nnz == 0) {
        // No connections - allocate minimal structure
        nnz = 1;
    }
    
    cudaError_t err = matrix.allocate(num_source, num_target, nnz);
    if (err != cudaSuccess) return err;
    
    if (col_idx.empty()) {
        // Initialize empty matrix
        std::vector<int> empty_row_ptr(num_target + 1, 0);
        err = cudaMemcpy(matrix.d_row_ptr, empty_row_ptr.data(), (num_target + 1) * sizeof(int), cudaMemcpyHostToDevice);
        return err;
    }
    
    err = cudaMemcpy(matrix.d_row_ptr, row_ptr.data(), (num_target + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(matrix.d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(matrix.d_weights, weights.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(matrix.d_eligibility, 0, nnz * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMemset(matrix.d_pre_trace, 0, num_source * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMemset(matrix.d_post_trace, 0, num_target * sizeof(float));
    
    return err;
}

/**
 * @brief Generate feedforward input connectivity to L4
 */
cudaError_t generateFeedforwardInputConnectivity(
    CSRSynapseMatrix& matrix,
    int input_size,
    int l4_neurons,
    float probability,
    float w_mean,
    unsigned int seed
) {
    return generateRandomConnectivity(matrix, input_size, l4_neurons, probability, w_mean, 0.02f, seed);
}

/**
 * @brief Generate output connectivity from L5
 */
cudaError_t generateOutputConnectivity(
    CSRSynapseMatrix& matrix,
    int l5_neurons,
    int output_size,
    float probability,
    float w_mean,
    unsigned int seed
) {
    return generateRandomConnectivity(matrix, l5_neurons, output_size, probability, w_mean, 0.02f, seed);
}

// ============================================================================
// GPU-ACCELERATED CONNECTIVITY GENERATION (for large networks)
// ============================================================================

/**
 * @brief GPU kernel to count connections per row
 */
__global__ void count_connections_kernel(
    int* row_counts,
    curandState* states,
    float probability,
    int num_pre,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    curandState local_state = states[row];
    int count = 0;
    
    for (int col = 0; col < num_pre; ++col) {
        if (curand_uniform(&local_state) < probability) {
            count++;
        }
    }
    
    row_counts[row] = count;
    states[row] = local_state;
}

/**
 * @brief GPU kernel to fill connectivity
 */
__global__ void fill_connectivity_kernel(
    const int* row_ptr,
    int* col_idx,
    float* weights,
    curandState* states,
    float probability,
    float w_mean,
    float w_std,
    int num_pre,
    int num_post
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_post) return;
    
    curandState local_state = states[row];
    int write_pos = row_ptr[row];
    
    for (int col = 0; col < num_pre; ++col) {
        if (curand_uniform(&local_state) < probability) {
            col_idx[write_pos] = col;
            float w = w_mean + w_std * curand_normal(&local_state);
            weights[write_pos] = fmaxf(0.0f, fminf(1.0f, w));
            write_pos++;
        }
    }
    
    states[row] = local_state;
}

/**
 * @brief Generate connectivity entirely on GPU (for large matrices)
 */
cudaError_t generateConnectivityGPU(
    CSRSynapseMatrix& matrix,
    int num_pre,
    int num_post,
    float probability,
    float w_mean,
    float w_std,
    unsigned long long seed,
    cudaStream_t stream
) {
    cudaError_t err;
    
    // Allocate random states
    curandState* d_states;
    err = cudaMalloc(&d_states, num_post * sizeof(curandState));
    if (err != cudaSuccess) return err;
    
    // Initialize random states
    int block_size = 256;
    int grid_size = (num_post + block_size - 1) / block_size;
    
    kernels::init_random_states_kernel<<<grid_size, block_size, 0, stream>>>(
        d_states, seed, num_post
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_states);
        return err;
    }
    
    // Count connections per row
    int* d_row_counts;
    err = cudaMalloc(&d_row_counts, num_post * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_states);
        return err;
    }
    
    count_connections_kernel<<<grid_size, block_size, 0, stream>>>(
        d_row_counts, d_states, probability, num_pre, num_post
    );
    
    // Compute row pointers using prefix sum
    int* d_row_ptr;
    err = cudaMalloc(&d_row_ptr, (num_post + 1) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_states);
        cudaFree(d_row_counts);
        return err;
    }
    
    // Use thrust for prefix sum
    thrust::device_ptr<int> counts_ptr(d_row_counts);
    thrust::device_ptr<int> row_ptr_ptr(d_row_ptr);
    
    // First element is 0
    cudaMemsetAsync(d_row_ptr, 0, sizeof(int), stream);
    
    // Exclusive scan
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
        counts_ptr, counts_ptr + num_post, row_ptr_ptr + 1);
    
    // Add first element to make it proper CSR row_ptr
    cudaStreamSynchronize(stream);
    
    // Get total nnz
    int nnz;
    cudaMemcpy(&nnz, d_row_ptr + num_post, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (nnz == 0) nnz = 1;  // Ensure at least minimal allocation
    
    // Allocate final arrays
    int* d_col_idx;
    float* d_weights;
    err = cudaMalloc(&d_col_idx, nnz * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_states);
        cudaFree(d_row_counts);
        cudaFree(d_row_ptr);
        return err;
    }
    
    err = cudaMalloc(&d_weights, nnz * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_states);
        cudaFree(d_row_counts);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        return err;
    }
    
    // Re-initialize random states for reproducibility
    kernels::init_random_states_kernel<<<grid_size, block_size, 0, stream>>>(
        d_states, seed, num_post
    );
    
    // Fill connectivity
    fill_connectivity_kernel<<<grid_size, block_size, 0, stream>>>(
        d_row_ptr, d_col_idx, d_weights, d_states,
        probability, w_mean, w_std, num_pre, num_post
    );
    
    // Set up matrix structure
    matrix.d_row_ptr = d_row_ptr;
    matrix.d_col_idx = d_col_idx;
    matrix.d_weights = d_weights;
    matrix.num_pre = num_pre;
    matrix.num_post = num_post;
    matrix.nnz = nnz;
    
    // Allocate plasticity arrays
    err = cudaMalloc(&matrix.d_eligibility, nnz * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&matrix.d_pre_trace, num_pre * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&matrix.d_post_trace, num_post * sizeof(float));
    if (err != cudaSuccess) return err;
    
    // Initialize to zero
    cudaMemsetAsync(matrix.d_eligibility, 0, nnz * sizeof(float), stream);
    cudaMemsetAsync(matrix.d_pre_trace, 0, num_pre * sizeof(float), stream);
    cudaMemsetAsync(matrix.d_post_trace, 0, num_post * sizeof(float), stream);
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_row_counts);
    
    return cudaSuccess;
}

} // namespace cortical
} // namespace neurogen
