/**
 * @file BioConnectivityGenerator.cu
 * @brief Topographic Connectivity Generator with Dale's Law
 *
 * This file implements biologically-realistic connectivity patterns:
 * - Topographic wiring based on spatial distance (Gaussian probability)
 * - Dale's Law enforcement (excitatory/inhibitory segregation)
 * - Receptive field generation for cortical processing
 *
 * @version 2.0-Biology-First
 * @date December 30, 2025
 */

#include "engine/NetworkConfig.h"
#include "engine/BioConstants.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

namespace neurogen {

// ============================================================================
// CUDA KERNEL: Topographic Connection Probability
// ============================================================================

/**
 * @brief Calculate connection probability based on 3D spatial distance
 *
 * Uses Gaussian probability distribution to create topographic connectivity:
 * - Nearby neurons have high connection probability
 * - Distant neurons have low connection probability
 * - Creates receptive fields and local processing columns
 */
__global__ void generate_topographic_mask_kernel(
    float* probability_matrix,
    int src_width, int src_height,
    int dst_width, int dst_height,
    float sigma, bool is_inhibitory)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_connections = (src_width * src_height) * (dst_width * dst_height);

    if (idx >= total_connections) return;

    // Decode linear index to (x1,y1) -> (x2,y2)
    int dst_neurons = dst_width * dst_height;
    int src_idx = idx / dst_neurons;
    int dst_idx = idx % dst_neurons;

    int s_x = src_idx % src_width;
    int s_y = src_idx / src_width;
    int d_x = dst_idx % dst_width;
    int d_y = dst_idx / dst_width;

    // Euclidean Distance
    float dist_sq = powf(float(s_x - d_x), 2.0f) + powf(float(s_y - d_y), 2.0f);

    // Gaussian Probability: P = e^(-dist^2 / (2*sigma^2))
    float prob = expf(-dist_sq / (2.0f * sigma * sigma));

    // Inhibitory neurons have tighter, denser connectivity (Lateral Inhibition)
    if (is_inhibitory) {
        if (dist_sq < 1.0f) {
            prob = 0.0f; // No self-inhibition
        } else {
            prob *= 1.5f; // Stronger local connection probability
        }
    }

    probability_matrix[idx] = prob;
}

// ============================================================================
// BioConnectivityGenerator Class
// ============================================================================

class BioConnectivityGenerator {
public:
    /**
     * @brief Generate cortical wiring with Dale's Law and topographic organization
     *
     * Creates a sparse connectivity matrix (CSR format) with:
     * - Excitatory neurons (first num_excitatory indices) have positive weights
     * - Inhibitory neurons (remaining indices) have negative weights
     * - Connection probability based on spatial distance
     * - No autapses (self-connections)
     *
     * @param row_ptr CSR row pointer array (output)
     * @param col_ind CSR column index array (output)
     * @param values Weight values array (output)
     * @param num_neurons Total number of neurons
     * @param width Spatial width of the layer
     * @param height Spatial height of the layer
     */
    static void generateCorticalWiring(
        std::vector<int>& row_ptr,
        std::vector<int>& col_ind,
        std::vector<float>& values,
        int num_neurons,
        int width, int height)
    {
        // Calculate neuron type boundary
        int num_excitatory = static_cast<int>(num_neurons * neurogen::bio::EXCITATORY_RATIO);
        int num_inhibitory = num_neurons - num_excitatory;

        std::cout << "🧬 Generating biological connectivity:" << std::endl;
        std::cout << "   Total neurons: " << num_neurons << std::endl;
        std::cout << "   Excitatory (Pyramidal): " << num_excitatory
                  << " (" << (neurogen::bio::EXCITATORY_RATIO * 100) << "%)" << std::endl;
        std::cout << "   Inhibitory (Interneuron): " << num_inhibitory
                  << " (" << (neurogen::bio::INHIBITORY_RATIO * 100) << "%)" << std::endl;

        // Split indices: [0..E-1] are Excitatory, [E..N-1] are Inhibitory
        // This sorting prevents warp divergence in CUDA kernels

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        row_ptr.clear();
        col_ind.clear();
        values.clear();

        row_ptr.push_back(0);

        int total_connections = 0;

        // Generate connections for each neuron
        for (int src = 0; src < num_neurons; src++) {
            bool is_inhibitory = (src >= num_excitatory);
            float sigma = is_inhibitory
                ? neurogen::bio::LATERAL_INHIBITION_RADIUS
                : neurogen::bio::LOCAL_SPREAD_SIGMA;

            // Get spatial coordinates of source neuron
            int s_x = src % width;
            int s_y = src / width;

            // Temporary storage for this neuron's connections
            std::vector<std::pair<int, float>> connections;

            for (int dst = 0; dst < num_neurons; dst++) {
                if (src == dst) continue; // No autapses

                int d_x = dst % width;
                int d_y = dst / width;

                // Calculate distance
                float distance_sq = pow(s_x - d_x, 2) + pow(s_y - d_y, 2);
                float prob = exp(-distance_sq / (2 * sigma * sigma));

                // Apply sparsity threshold
                float connection_prob = prob * neurogen::bio::MAX_CONNECTION_DENSITY;

                if (is_inhibitory) {
                    // Inhibitory: no self-inhibition, but stronger local connections
                    if (distance_sq < 1.0f) {
                        connection_prob = 0.0f;
                    } else {
                        connection_prob *= 1.5f;
                    }
                }

                // Stochastic connection based on distance
                if (dist(gen) < connection_prob) {
                    // Generate initial weight
                    float weight = dist(gen);

                    // DALE'S LAW ENFORCEMENT
                    if (is_inhibitory) {
                        // Inhibition is strong and negative
                        weight = -weight * neurogen::bio::INHIBITORY_STRENGTH_MULTIPLIER;
                        weight = neurogen::bio::clampWeightDalesLaw(weight, true);
                    } else {
                        // Excitation is positive
                        weight = neurogen::bio::clampWeightDalesLaw(weight, false);
                    }

                    connections.push_back({dst, weight});
                }
            }

            // Add connections to CSR format
            for (const auto& conn : connections) {
                col_ind.push_back(conn.first);
                values.push_back(conn.second);
                total_connections++;
            }

            row_ptr.push_back(col_ind.size());
        }

        std::cout << "   Total connections: " << total_connections
                  << " (density: " << (100.0f * total_connections / (num_neurons * num_neurons))
                  << "%)" << std::endl;

        // Statistics
        int exc_connections = 0;
        int inh_connections = 0;
        for (int src = 0; src < num_neurons; src++) {
            int start = row_ptr[src];
            int end = row_ptr[src + 1];
            if (src < num_excitatory) {
                exc_connections += (end - start);
            } else {
                inh_connections += (end - start);
            }
        }

        std::cout << "   Excitatory connections: " << exc_connections << std::endl;
        std::cout << "   Inhibitory connections: " << inh_connections << std::endl;
        std::cout << "   Avg fanout (excitatory): "
                  << (num_excitatory > 0 ? exc_connections / num_excitatory : 0) << std::endl;
        std::cout << "   Avg fanout (inhibitory): "
                  << (num_inhibitory > 0 ? inh_connections / num_inhibitory : 0) << std::endl;
    }

    /**
     * @brief Generate connectivity with custom neuron type parameters
     *
     * Advanced version that allows per-neuron type specification
     */
    static void generateCorticalWiringAdvanced(
        std::vector<int>& row_ptr,
        std::vector<int>& col_ind,
        std::vector<float>& values,
        const std::vector<neurogen::bio::NeuronType>& neuron_types,
        int width, int height)
    {
        int num_neurons = neuron_types.size();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        row_ptr.clear();
        col_ind.clear();
        values.clear();

        row_ptr.push_back(0);

        for (int src = 0; src < num_neurons; src++) {
            bool is_inhibitory = (neuron_types[src] == neurogen::bio::NeuronType::INHIBITORY);
            float sigma = is_inhibitory
                ? neurogen::bio::LATERAL_INHIBITION_RADIUS
                : neurogen::bio::LOCAL_SPREAD_SIGMA;

            int s_x = src % width;
            int s_y = src / width;

            for (int dst = 0; dst < num_neurons; dst++) {
                if (src == dst) continue;

                int d_x = dst % width;
                int d_y = dst / width;

                float distance_sq = pow(s_x - d_x, 2) + pow(s_y - d_y, 2);
                float prob = exp(-distance_sq / (2 * sigma * sigma));
                float connection_prob = prob * neurogen::bio::MAX_CONNECTION_DENSITY;

                if (is_inhibitory && distance_sq < 1.0f) {
                    connection_prob = 0.0f;
                } else if (is_inhibitory) {
                    connection_prob *= 1.5f;
                }

                if (dist(gen) < connection_prob) {
                    float weight = dist(gen);

                    if (is_inhibitory) {
                        weight = -weight * neurogen::bio::INHIBITORY_STRENGTH_MULTIPLIER;
                        weight = neurogen::bio::clampWeightDalesLaw(weight, true);
                    } else {
                        weight = neurogen::bio::clampWeightDalesLaw(weight, false);
                    }

                    col_ind.push_back(dst);
                    values.push_back(weight);
                }
            }

            row_ptr.push_back(col_ind.size());
        }
    }
};

// Wrapper function for external linkage
void generateCorticalWiring(
    std::vector<int>& row_ptr,
    std::vector<int>& col_ind,
    std::vector<float>& values,
    int num_neurons,
    int width, int height)
{
    BioConnectivityGenerator::generateCorticalWiring(row_ptr, col_ind, values, num_neurons, width, height);
}

} // namespace neurogen
