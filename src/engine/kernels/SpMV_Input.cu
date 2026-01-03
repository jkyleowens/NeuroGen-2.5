#include "engine/kernels/SpMV_Input.cuh"
#include <cuda_runtime.h>

namespace neurogen {
namespace kernels {

// ============================================================================
// LEGACY BINARY SPIKE SUPPORT
// ============================================================================

// Conversion kernel for legacy uint8_t spikes
__global__ void castSpikesToFloatKernel(const uint8_t* spikes, float* float_spikes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float_spikes[idx] = (float)spikes[idx];
    }
}

void computeSynapticInputs(
    neurogen::SparseMatrix& matrix,
    const uint8_t* d_spikes,
    float* d_input_current,
    float* d_temp_float_spikes,
    int num_neurons,
    cudaStream_t stream)
{
    // 1. Convert uint8 spikes to float for cuSPARSE compatibility
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;

    castSpikesToFloatKernel<<<blocks, threads, 0, stream>>>(
        d_spikes, d_temp_float_spikes, num_neurons
    );

    // 2. Perform SpMV
    // Input_Current += Weights * Float_Spikes
    // alpha=1.0 (add to existing input), beta=1.0 (preserve existing input accumulator)
    matrix.vectorMultiply(d_temp_float_spikes, d_input_current, 1.0f, 1.0f);
}

// ============================================================================
// WEIGHTED SPIKE OUTPUT SUPPORT (New API)
// ============================================================================

/**
 * @brief Computes synaptic inputs from weighted (float) spikes
 *
 * This is the preferred method when using the weighted spike output model.
 * It directly uses float spike amplitudes without conversion overhead.
 *
 * The SpMV operation computes:
 *   I_post += W * S_weighted
 *
 * Where:
 *   - W is the sparse synaptic weight matrix
 *   - S_weighted is the weighted spike output vector (floats)
 *   - I_post is the postsynaptic current accumulator
 *
 * Benefits over legacy API:
 *   - No memory allocation for temporary buffer
 *   - No conversion kernel overhead
 *   - Spike amplitudes are preserved in synaptic transmission
 *   - More biologically realistic (stronger spikes = more current)
 */
void computeSynapticInputsWeighted(
    neurogen::SparseMatrix& matrix,
    const float* d_spike_output,
    float* d_input_current,
    int num_neurons,
    cudaStream_t stream)
{
    // Direct SpMV with weighted spike outputs
    // Input_Current += Weights * WeightedSpikes
    // alpha=1.0 (scale factor for SpMV result)
    // beta=1.0 (preserve existing input accumulator)
    //
    // The weighted spike output already contains the amplitude information,
    // so the synaptic current contribution is:
    //   I_synapse = sum(w_ij * spike_amplitude_j)
    //
    // This naturally implements:
    // - Stronger spikes → more current (synaptic facilitation analog)
    // - Sparse operation (zero spikes contribute nothing)
    // - Efficient GPU utilization (native float SpMV)

    matrix.vectorMultiply(d_spike_output, d_input_current, 1.0f, 1.0f);

    // Note: Stream parameter not yet passed to SparseMatrix::vectorMultiply
    // TODO: Update SparseMatrix interface to accept cudaStream_t for full async
    (void)stream;  // Suppress unused parameter warning
    (void)num_neurons;  // Used for documentation, not in SpMV call
}

} // namespace kernels
} // namespace neurogen


