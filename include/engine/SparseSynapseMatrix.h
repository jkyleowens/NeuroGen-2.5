/**
 * @file SparseSynapseMatrix.h
 * @brief Block Sparse Row (BSR) Synapse Matrix for efficient GPU computation
 * 
 * This file defines the new sparse synapse model for NeuroGen 2.0's cortical column
 * architecture. Key features:
 * - Block Sparse Row (BSR) format optimized for warp-level operations
 * - 32x32 block size for tensor core compatibility (future)
 * - Efficient SpMV operations for synaptic propagation
 * - Dynamic connectivity support (synaptogenesis/pruning)
 * 
 * Memory comparison:
 * - Old: 40 bytes per synapse (AoS struct)
 * - New: 8 bytes per synapse (weight + 4-byte overhead amortized)
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef SPARSE_SYNAPSE_MATRIX_H
#define SPARSE_SYNAPSE_MATRIX_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdint>
#include <vector>

namespace neurogen {
namespace cortical {

// ============================================================================
// SPARSE MATRIX FORMATS
// ============================================================================

/**
 * @brief Compressed Sparse Row (CSR) format for synaptic connectivity
 * 
 * Standard CSR format:
 * - row_ptr[i] to row_ptr[i+1] gives range of column indices for row i
 * - col_idx contains column indices
 * - values contains weights
 * 
 * For synapses: rows = presynaptic neurons, cols = postsynaptic neurons
 */
struct CSRSynapseMatrix {
    // === CSR STRUCTURE ===
    int* d_row_ptr;             // [num_pre + 1] Row pointers
    int* d_col_idx;             // [nnz] Column indices (post neuron)
    float* d_weights;           // [nnz] Synaptic weights
    
    // === PLASTICITY DATA ===
    float* d_eligibility;       // [nnz] Eligibility traces
    float* d_pre_trace;         // [num_pre] Presynaptic activity trace
    float* d_post_trace;        // [num_post] Postsynaptic activity trace
    
    // === SYNAPSE PROPERTIES ===
    float* d_delays;            // [nnz] Synaptic delays (optional)
    int8_t* d_synapse_type;     // [nnz] 0=excitatory, 1=inhibitory
    
    // === DIMENSIONS ===
    int num_pre;                // Number of presynaptic neurons (rows)
    int num_post;               // Number of postsynaptic neurons (cols)
    int nnz;                    // Number of non-zero elements (synapses)
    
    // === CUSPARSE HANDLE ===
    cusparseSpMatDescr_t descr; // cuSPARSE matrix descriptor
    
    // === MEMORY MANAGEMENT ===
    
    cudaError_t allocate(int n_pre, int n_post, int n_synapses) {
        num_pre = n_pre;
        num_post = n_post;
        nnz = n_synapses;
        cudaError_t err;
        
        // CSR structure
        err = cudaMalloc(&d_row_ptr, (n_pre + 1) * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_col_idx, n_synapses * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_weights, n_synapses * sizeof(float));
        if (err != cudaSuccess) return err;
        
        // Plasticity
        err = cudaMalloc(&d_eligibility, n_synapses * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_pre_trace, n_pre * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_post_trace, n_post * sizeof(float));
        if (err != cudaSuccess) return err;
        
        // Properties (optional, allocate on demand)
        d_delays = nullptr;
        d_synapse_type = nullptr;
        
        return cudaSuccess;
    }
    
    void free() {
        if (d_row_ptr) cudaFree(d_row_ptr);
        if (d_col_idx) cudaFree(d_col_idx);
        if (d_weights) cudaFree(d_weights);
        if (d_eligibility) cudaFree(d_eligibility);
        if (d_pre_trace) cudaFree(d_pre_trace);
        if (d_post_trace) cudaFree(d_post_trace);
        if (d_delays) cudaFree(d_delays);
        if (d_synapse_type) cudaFree(d_synapse_type);
        
        d_row_ptr = d_col_idx = nullptr;
        d_weights = d_eligibility = nullptr;
        d_pre_trace = d_post_trace = nullptr;
        d_delays = nullptr;
        d_synapse_type = nullptr;
    }
    
    size_t getMemoryFootprint() const {
        // row_ptr + col_idx + weights + eligibility + traces
        return (num_pre + 1) * 4 + nnz * 4 + nnz * 4 + nnz * 4 
               + num_pre * 4 + num_post * 4;
    }
};

/**
 * @brief Block Sparse Row (BSR) format for warp-optimized operations
 * 
 * BSR groups synapses into blocks of size BLOCK_SIZE x BLOCK_SIZE.
 * Each block is stored densely, allowing efficient warp-level operations.
 * 
 * For GTX 1650: BLOCK_SIZE=32 aligns with warp size for optimal performance.
 */
template<int BLOCK_SIZE = 32>
struct BSRSynapseMatrix {
    // === BSR STRUCTURE ===
    int* d_block_row_ptr;       // [num_block_rows + 1] Block row pointers
    int* d_block_col_idx;       // [num_blocks] Block column indices
    float* d_block_values;      // [num_blocks * BLOCK_SIZE * BLOCK_SIZE] Dense blocks
    
    // === BLOCK-LEVEL PLASTICITY ===
    float* d_block_eligibility; // [num_blocks * BLOCK_SIZE * BLOCK_SIZE] Eligibility per synapse
    float* d_block_mask;        // [num_blocks * BLOCK_SIZE * BLOCK_SIZE] Active synapse mask
    
    // === TRACES (per neuron, not per synapse) ===
    float* d_pre_trace;         // [num_pre] Presynaptic traces
    float* d_post_trace;        // [num_post] Postsynaptic traces
    
    // === DIMENSIONS ===
    int num_pre;                // Total presynaptic neurons
    int num_post;               // Total postsynaptic neurons
    int num_block_rows;         // = ceil(num_pre / BLOCK_SIZE)
    int num_block_cols;         // = ceil(num_post / BLOCK_SIZE)
    int num_blocks;             // Number of non-zero blocks
    int nnz;                    // Actual number of synapses (including zeros in blocks)
    
    // === CUSPARSE HANDLE ===
    cusparseSpMatDescr_t descr;
    
    static constexpr int block_size = BLOCK_SIZE;
    
    cudaError_t allocate(int n_pre, int n_post, int n_blocks) {
        num_pre = n_pre;
        num_post = n_post;
        num_block_rows = (n_pre + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_block_cols = (n_post + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_blocks = n_blocks;
        nnz = n_blocks * BLOCK_SIZE * BLOCK_SIZE;
        
        cudaError_t err;
        
        // BSR structure
        err = cudaMalloc(&d_block_row_ptr, (num_block_rows + 1) * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_block_col_idx, n_blocks * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_block_values, nnz * sizeof(float));
        if (err != cudaSuccess) return err;
        
        // Plasticity
        err = cudaMalloc(&d_block_eligibility, nnz * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_block_mask, nnz * sizeof(float));
        if (err != cudaSuccess) return err;
        
        // Traces
        err = cudaMalloc(&d_pre_trace, n_pre * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_post_trace, n_post * sizeof(float));
        if (err != cudaSuccess) return err;
        
        return cudaSuccess;
    }
    
    void free() {
        if (d_block_row_ptr) cudaFree(d_block_row_ptr);
        if (d_block_col_idx) cudaFree(d_block_col_idx);
        if (d_block_values) cudaFree(d_block_values);
        if (d_block_eligibility) cudaFree(d_block_eligibility);
        if (d_block_mask) cudaFree(d_block_mask);
        if (d_pre_trace) cudaFree(d_pre_trace);
        if (d_post_trace) cudaFree(d_post_trace);
        
        d_block_row_ptr = d_block_col_idx = nullptr;
        d_block_values = d_block_eligibility = d_block_mask = nullptr;
        d_pre_trace = d_post_trace = nullptr;
    }
    
    size_t getMemoryFootprint() const {
        return (num_block_rows + 1) * 4 + num_blocks * 4 + nnz * 4 * 3
               + num_pre * 4 + num_post * 4;
    }
};

// ============================================================================
// SYNAPSE PROPAGATION PARAMETERS
// ============================================================================

/**
 * @brief Parameters for synaptic transmission
 */
struct SynapseParameters {
    // Time constants
    float tau_rise_exc = 0.5f;      // Excitatory rise time (ms)
    float tau_decay_exc = 2.0f;     // Excitatory decay time (ms)
    float tau_rise_inh = 0.5f;      // Inhibitory rise time (ms)
    float tau_decay_inh = 5.0f;     // Inhibitory decay time (ms)
    
    // Reversal potentials
    float E_exc = 0.0f;             // Excitatory reversal (mV)
    float E_inh = -70.0f;           // Inhibitory reversal (mV)
    
    // Weight bounds
    float w_min = 0.0f;             // Minimum weight
    float w_max = 1.0f;             // Maximum weight
    
    // Plasticity
    float stdp_tau_pre = 20.0f;     // Pre-spike trace decay (ms)
    float stdp_tau_post = 20.0f;    // Post-spike trace decay (ms)
    float stdp_a_plus = 0.01f;      // LTP amplitude
    float stdp_a_minus = 0.012f;    // LTD amplitude (slightly stronger)
    
    // Eligibility trace
    float tau_eligibility = 1000.0f; // Eligibility trace decay (ms)
};

// ============================================================================
// CONNECTIVITY PATTERNS
// ============================================================================

/**
 * @brief Connectivity pattern types for cortical columns
 */
enum class ConnectivityPattern {
    RANDOM,                 // Random with probability p
    SMALL_WORLD,           // Small-world network (Watts-Strogatz)
    DISTANCE_DEPENDENT,    // Connection probability decays with distance
    LAYER_SPECIFIC,        // Based on cortical layer connectivity rules
    RICH_CLUB,             // Hub neurons with high connectivity
};

/**
 * @brief Parameters for connectivity generation
 */
struct ConnectivityParams {
    ConnectivityPattern pattern = ConnectivityPattern::SMALL_WORLD;
    float connection_probability = 0.1f;    // Base connection probability
    float rewiring_probability = 0.1f;      // Small-world rewiring
    float distance_scale = 100.0f;          // Distance scaling (µm)
    int k_nearest = 10;                     // k-nearest neighbors
    float hub_threshold = 0.9f;             // Rich club threshold
};

// ============================================================================
// SPARSE MATRIX OPERATIONS
// ============================================================================

/**
 * @brief Initialize cuSPARSE descriptors for a CSR matrix
 */
cusparseStatus_t initializeCuSparseCSR(
    CSRSynapseMatrix& matrix,
    cusparseHandle_t handle
);

/**
 * @brief Initialize cuSPARSE descriptors for a BSR matrix
 */
template<int BLOCK_SIZE>
cusparseStatus_t initializeCuSparseBSR(
    BSRSynapseMatrix<BLOCK_SIZE>& matrix,
    cusparseHandle_t handle
);

/**
 * @brief Perform sparse matrix-vector multiplication (synaptic propagation)
 *
 * y = alpha * A * x + beta * y
 * where A = weight matrix, x = presynaptic spikes, y = postsynaptic currents
 *
 * Legacy version for binary (uint8_t) spikes.
 */
cudaError_t sparseSynapticPropagation(
    const CSRSynapseMatrix& synapses,
    const uint8_t* d_pre_spikes,    // Presynaptic spike vector (binary)
    float* d_post_currents,          // Postsynaptic current accumulator
    float scale,                     // Scaling factor
    cusparseHandle_t handle,
    cudaStream_t stream = 0
);

/**
 * @brief Perform sparse matrix-vector multiplication with weighted spikes
 *
 * y = alpha * A * x + beta * y
 * where A = weight matrix, x = weighted spike outputs, y = postsynaptic currents
 *
 * === WEIGHTED SPIKE OUTPUT SUPPORT ===
 * This version accepts float spike outputs directly, enabling:
 * - Spike amplitude to influence synaptic current magnitude
 * - No conversion overhead from uint8_t to float
 * - More biologically realistic transmission (stronger spikes = more current)
 */
cudaError_t sparseSynapticPropagationWeighted(
    const CSRSynapseMatrix& synapses,
    const float* d_pre_spike_output,  // Presynaptic weighted spike outputs
    float* d_post_currents,            // Postsynaptic current accumulator
    float scale,                       // Scaling factor
    cusparseHandle_t handle,
    cudaStream_t stream = 0
);

/**
 * @brief Block-sparse synaptic propagation (warp-optimized) - binary spikes
 */
template<int BLOCK_SIZE>
cudaError_t blockSparseSynapticPropagation(
    const BSRSynapseMatrix<BLOCK_SIZE>& synapses,
    const uint8_t* d_pre_spikes,
    float* d_post_currents,
    float scale,
    cudaStream_t stream = 0
);

/**
 * @brief Block-sparse synaptic propagation with weighted spikes
 */
template<int BLOCK_SIZE>
cudaError_t blockSparseSynapticPropagationWeighted(
    const BSRSynapseMatrix<BLOCK_SIZE>& synapses,
    const float* d_pre_spike_output,  // Weighted spike outputs
    float* d_post_currents,
    float scale,
    cudaStream_t stream = 0
);

/**
 * @brief Generate sparse connectivity pattern
 */
cudaError_t generateConnectivity(
    CSRSynapseMatrix& synapses,
    int num_pre,
    int num_post,
    const ConnectivityParams& params,
    unsigned int seed = 42
);

/**
 * @brief Convert CSR to BSR format
 */
template<int BLOCK_SIZE>
cudaError_t convertCSRtoBSR(
    const CSRSynapseMatrix& csr,
    BSRSynapseMatrix<BLOCK_SIZE>& bsr
);

} // namespace cortical
} // namespace neurogen

#endif // SPARSE_SYNAPSE_MATRIX_H
