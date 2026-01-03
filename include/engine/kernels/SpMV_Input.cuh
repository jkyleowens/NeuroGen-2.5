#pragma once
#include "engine/SparseMatrix.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace neurogen {
namespace kernels {

/**
 * @brief Computes synaptic inputs using Sparse Matrix-Vector Multiplication.
 *
 * === WEIGHTED SPIKE OUTPUT SUPPORT ===
 * This module now supports float-valued spike outputs directly, eliminating
 * the need for uint8_t to float conversion when using weighted spikes.
 *
 * Two versions are provided:
 * 1. computeSynapticInputs (legacy): Takes uint8_t spikes, converts to float
 * 2. computeSynapticInputsWeighted: Takes float spikes directly (more efficient)
 *
 * For the weighted spike model, use computeSynapticInputsWeighted as it:
 * - Avoids memory allocation for temporary buffer
 * - Eliminates conversion kernel overhead
 * - Directly uses weighted spike amplitudes in SpMV
 */

/**
 * @brief Computes synaptic inputs from binary (uint8_t) spikes (legacy API)
 *
 * Optimized for neural network connectivity:
 * Input Current += WeightMatrix * Spikes
 *
 * @param matrix The sparse weight matrix (BSR/CSR)
 * @param d_spikes Spike array (uint8_t) from neurons - binary 0/1
 * @param d_input_current Output current accumulator
 * @param d_temp_float_spikes Temporary buffer for float conversion
 * @param num_neurons Total neurons
 * @param stream CUDA stream
 */
void computeSynapticInputs(
    neurogen::SparseMatrix& matrix,
    const uint8_t* d_spikes,
    float* d_input_current,
    float* d_temp_float_spikes,
    int num_neurons,
    cudaStream_t stream
);

/**
 * @brief Computes synaptic inputs from weighted (float) spikes
 *
 * Optimized for weighted spike output model:
 * Input Current += WeightMatrix * WeightedSpikes
 *
 * This version is more efficient when using float spike outputs because:
 * - No conversion from uint8_t to float needed
 * - No temporary buffer allocation required
 * - Weighted spikes carry amplitude information directly
 *
 * @param matrix The sparse weight matrix (BSR/CSR)
 * @param d_spike_output Weighted spike outputs (float) from neurons
 * @param d_input_current Output current accumulator
 * @param num_neurons Total neurons
 * @param stream CUDA stream
 */
void computeSynapticInputsWeighted(
    neurogen::SparseMatrix& matrix,
    const float* d_spike_output,
    float* d_input_current,
    int num_neurons,
    cudaStream_t stream
);

} // namespace kernels
} // namespace neurogen
