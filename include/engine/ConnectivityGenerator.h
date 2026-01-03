/**
 * @file ConnectivityGenerator.h
 * @brief Connectivity pattern generation for cortical columns
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef CONNECTIVITY_GENERATOR_H
#define CONNECTIVITY_GENERATOR_H

#include "SparseSynapseMatrix.h"
#include <vector>
#include <cuda_runtime.h>

namespace neurogen {
namespace cortical {

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
    unsigned int seed = 42
);

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
    unsigned int seed = 42
);

/**
 * @brief Generate distance-dependent connectivity
 */
cudaError_t generateDistanceDependentConnectivity(
    CSRSynapseMatrix& matrix,
    int num_pre,
    int num_post,
    const std::vector<float>& pre_positions,
    const std::vector<float>& post_positions,
    int dims,
    float sigma,
    float p_base,
    float w_mean,
    float w_std,
    unsigned int seed = 42
);

/**
 * @brief Generate canonical cortical connectivity between layers
 */
cudaError_t generateInterLayerConnectivity(
    CSRSynapseMatrix& matrix,
    int source_layer,
    int target_layer,
    int num_source,
    int num_target,
    float exc_ratio = 0.8f,
    float w_exc_mean = 0.1f,
    float w_inh_mean = 0.15f,
    unsigned int seed = 42
);

/**
 * @brief Generate feedforward input connectivity to L4
 */
cudaError_t generateFeedforwardInputConnectivity(
    CSRSynapseMatrix& matrix,
    int input_size,
    int l4_neurons,
    float probability = 0.1f,
    float w_mean = 0.2f,
    unsigned int seed = 42
);

/**
 * @brief Generate output connectivity from L5
 */
cudaError_t generateOutputConnectivity(
    CSRSynapseMatrix& matrix,
    int l5_neurons,
    int output_size,
    float probability = 0.1f,
    float w_mean = 0.2f,
    unsigned int seed = 42
);

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
    unsigned long long seed = 42,
    cudaStream_t stream = 0
);

} // namespace cortical
} // namespace neurogen

#endif // CONNECTIVITY_GENERATOR_H
