#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace neurogen {
namespace kernels {

struct LIFParams {
    float alpha;        // Voltage decay (e.g. 0.95)
    float beta;         // Adaptation decay (e.g. 0.90)
    float delta;        // Adaptation increment (e.g. 1.0)
    float v_thresh;     // Firing threshold (e.g. -50.0)
    float v_reset;      // Reset potential (e.g. -65.0)
    int k_winners;      // For kWTA (e.g. 25 for 10% sparsity in 256 block)
};

/**
 * @brief Fused LIF-A update kernel with intra-block kWTA inhibition.
 * 
 * Performs:
 * 1. Integration: V = alpha*V + I - A
 * 2. Adaptation: A = beta*A
 * 3. Local Inhibition (kWTA): Only top k neurons in block are allowed to spike
 * 4. Spike Generation & Reset
 * 
 * @param d_v Voltage array
 * @param d_a Adaptation array
 * @param d_spikes Spike bitmask (output)
 * @param d_input Input current (accumulated)
 * @param d_last_spike_time Last spike timestamp
 * @param current_time Current simulation time
 * @param num_neurons Total neurons
 * @param dt Time step
 * @param params Neuron parameters
 */
void launchLIFUpdate(
    float* d_v, 
    float* d_a, 
    uint8_t* d_spikes, 
    const float* d_input,
    float* d_last_spike_time,
    float current_time,
    int num_neurons,
    float dt,
    LIFParams params,
    cudaStream_t stream
);

} // namespace kernels
} // namespace neurogen

